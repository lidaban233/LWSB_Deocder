import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from helper_functions_LDPC_torch import load_code, syndrome
from tqdm import tqdm
import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


DEBUG = False
TRAINING = True
FROZEN_TRAINING = True
TRAING_WISE_LAYER_NOISE = True
FROZEN_TRAINING_alpha = 0.95
SUM_PRODUCT = False  # Sum-Product Algorithm (SPA)
MIN_SUM = not SUM_PRODUCT  # Min-Sum Algorithm (MSA)
ALL_ZEROS_CODEWORD_TRAINING = False
ALL_ZEROS_CODEWORD_TESTING = False
NO_SIGMA_SCALING_TRAIN = False
NO_SIGMA_SCALING_TEST = False
np.set_printoptions(precision=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using PyTorch version:", torch.__version__)

# Define parameters
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
# 加入time时间戳
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
log_path = 'train_log_' + time_stamp + '.txt'
print(log_path)
test_layers = [i for i in range(0, 10)]
learning_rate = 0.0005
snr_lo = -2.0
snr_hi = 5.0
snr_step = 1.0
min_frame_errors = 0 # 1000
min_frame = 0 # 100
max_frames = 0 # 100000
num_iterations = 50
H_filename = 'codes/LDPC_chk_mat_270_150.txt'
G_filename = 'codes/LDPC_gen_mat_270_120.txt'
L = 0.5
steps = 200
batch_size = 288
save_interval = 50
save_path = "./pytorch_checkpoints_loss-SFT-f10"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load LDPC code
code = load_code(H_filename, G_filename)
H = torch.tensor(code.H, dtype=torch.float32)
G = torch.tensor(code.G.T, dtype=torch.float32)
var_degrees = code.var_degrees
chk_degrees = code.chk_degrees
num_edges = sum(code.num_edges)
u = code.u
d = code.d
n = code.n
m = code.m
k = code.k


def set_requires_grad(model, test_layers):
    """
    根据给定的test_layers列表来设置哪些B_cv行需要计算梯度。
    :param model: 模型
    :param test_layers: 需要计算梯度的层的索引列表
    """
    with torch.no_grad():  # 禁用梯度计算
        for i in range(model.B_cv.shape[0]):
            if i in test_layers:
                model.B_cv[i].requires_grad = True  # 如果是需要训练的层，开启梯度计算
            else:
                model.B_cv[i].requires_grad = False  # 否则冻结该层


def truncated_normal(tensor, mean=0.0, std=1.0):
    torch.nn.init.trunc_normal_(tensor, mean=mean, std=std)
    return tensor


class Decoder(nn.Module):
    def __init__(self, decoder_type="FNOMS", num_edges=0, num_iterations=50, relaxed=False):
        super(Decoder, self).__init__()
        self.decoder_type = decoder_type
        self.num_iterations = num_iterations
        self.relaxed = relaxed
        # 根据 decoder_type 定义参数
        if decoder_type == "FNSPA":
            self.W_cv = nn.Parameter(truncated_normal(torch.empty(num_iterations, num_edges), mean=0.0, std=1.0))
        elif decoder_type == "RNSPA":
            self.W_cv = nn.Parameter(truncated_normal(torch.empty(num_edges), mean=0.0, std=1.0))
        elif decoder_type == "FNNMS":
            self.W_cv = nn.Parameter(truncated_normal(torch.empty(num_iterations, num_edges), mean=0.0, std=1.0))
        elif decoder_type == "RNNMS":
            self.W_cv = nn.Parameter(
                torch.nn.functional.softplus(truncated_normal(torch.empty(num_edges), mean=0.0, std=1.0)))
        elif decoder_type == "FNOMS":
            self.B_cv = nn.Parameter(truncated_normal(torch.empty(num_iterations, num_edges), mean=0.0, std=1.0))
        elif decoder_type == "RNOMS":
            self.B_cv = nn.Parameter(truncated_normal(torch.empty(num_edges), mean=0.0, std=1.0))
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")
        self.relaxation_factors = nn.Parameter(torch.zeros(1)) if relaxed else None

    def forward(self, soft_input, labels, code):
        loss = 0.0
        cv = torch.zeros(num_edges, batch_size, dtype=torch.float32, device=device)
        vc = soft_input.clone()
        m_t = cv.clone() if self.relaxed else None

        for iteration in range(self.num_iterations):
            vc = self.compute_vc(cv, soft_input, code)
            if self.relaxed:
                R = torch.sigmoid(self.relaxation_factors)
                m_t = R * m_t + (1 - R) * vc
                vc = m_t
            cv = self.compute_cv(vc, iteration, code)
            soft_output = self.marginalize(soft_input, cv, code)
            loss += self.compute_loss(soft_output, labels, code)
        # print('soft_output', soft_output)
        # print('labels', labels)
        return loss, soft_output

    def compute_vc(self, cv, soft_input, code):
        # 初始化
        edges = []
        for i in range(code.n):
            for j in range(code.var_degrees[i]):
                edges.append(i)

        # 将 soft_input 按照 edges 重排
        reordered_soft_input = soft_input[edges]

        num_vc_elements = sum(code.var_degrees)  # 计算 vc 元素的总数
        vc = torch.zeros(num_vc_elements, batch_size, dtype=torch.float32, device=device)
        edge_order = []
        index = 0

        # 遍历每个变量节点
        for i in range(code.n):  # for each variable node
            for j in range(code.var_degrees[i]):
                # 当前边索引
                edge_order.append(code.d[i][j])

                # 获取当前变量节点的外部边
                extrinsic_edges = []
                for jj in range(code.var_degrees[i]):
                    if jj != j:  # 排除自身vc
                        extrinsic_edges.append(code.d[i][jj])

                # 计算外部边的消息
                if extrinsic_edges:
                    temp = cv[extrinsic_edges].sum(dim=0)  # 外部边的消息求和
                else:
                    temp = torch.zeros(cv.size(1), dtype=cv.dtype, device=device)  # 无外部边则置 0

                vc[index] = temp
                index += 1

        # 按照边的顺序重新排列
        new_order = torch.zeros(num_edges, dtype=torch.int64, device=device)
        new_order[edge_order] = torch.arange(0, num_edges, dtype=torch.int64, device=device)
        vc = vc[new_order]

        # vc 加上重新排列的 soft_input
        vc = vc + reordered_soft_input

        return vc

    def compute_cv(self, vc, iteration, code):
        cv_list = []
        prod_list = []
        min_list = []

        if SUM_PRODUCT:
            vc = torch.clamp(vc, -10, 10)  # 限制 vc 的范围
            tanh_vc = torch.tanh(vc / 2.0)  # 对 vc 进行 tanh 变换

        edge_order = []

        # 遍历每个检查节点
        for i in range(code.m):  # for each check node
            for j in range(code.chk_degrees[i]):
                # 当前边的索引
                edge_order.append(code.u[i][j])

                # 获取检查节点的外部边
                extrinsic_edges = [code.u[i][jj] for jj in range(code.chk_degrees[i]) if jj != j]

                if SUM_PRODUCT:
                    # SUM_PRODUCT 计算逻辑
                    temp = tanh_vc[extrinsic_edges].prod(dim=0)
                    temp = torch.log((1 + temp) / (1 - temp))
                    cv_list.append(temp)
                elif MIN_SUM:
                    # MIN_SUM 计算逻辑
                    temp = vc[extrinsic_edges]
                    temp1 = torch.sign(temp).prod(dim=0)
                    temp2 = torch.abs(temp).min(dim=0).values
                    prod_list.append(temp1)
                    min_list.append(temp2)

        if SUM_PRODUCT:
            # 汇总 SUM_PRODUCT 结果
            cv = torch.stack(cv_list).to(device)
        elif MIN_SUM:
            # 汇总 MIN_SUM 结果
            prods = torch.stack(prod_list).to(device)
            mins = torch.stack(min_list).to(device)
            if self.decoder_type == "RNOMS":
                mins = torch.relu(mins - self.B_cv.to(device))
            elif self.decoder_type == "FNOMS":
                offset = torch.nn.functional.softplus(self.B_cv[iteration].to(device))
                offset = offset.unsqueeze(1).repeat(1, batch_size)
                mins = torch.relu(mins - offset)
            cv = prods * mins


        # 重新排列 cv 的顺序
        new_order = torch.zeros(num_edges, dtype=torch.int64, device=device)
        new_order[edge_order] = torch.arange(num_edges, dtype=torch.int64, device=device)
        cv = cv[new_order]

        # 应用权重调整
        if self.decoder_type in ["RNSPA", "RNNMS"]:
            cv = cv * self.W_cv.unsqueeze(1).expand(-1, batch_size, -1)
        elif self.decoder_type in ["FNSPA", "FNNMS"]:
            cv = cv * self.W_cv[iteration].unsqueeze(1).expand(-1, batch_size, -1)

        return cv

    def marginalize(self, soft_input, cv, code):
        soft_output = soft_input.clone()
        for i in range(len(code.var_degrees)):
            edges = [code.d[i][j] for j in range(code.var_degrees[i])]
            soft_output[i] += cv[edges].sum(0)
        return soft_output

    def compute_loss(self, soft_output, labels, code):
        cross_entropy_loss = nn.functional.binary_cross_entropy_with_logits(-soft_output, labels) / num_iterations
        syndrome_loss = torch.max(1.0 - syndrome(soft_output, code), torch.tensor(0.0)).mean() / num_iterations
        # print('CE_loss', cross_entropy_loss)
        # print('syndrome_loss', syndrome_loss)
        # print()
        return L * cross_entropy_loss + (1 - L) * syndrome_loss


# Instantiate the decoder
decoder = Decoder(decoder_type="FNOMS", num_edges=num_edges, num_iterations=num_iterations, relaxed=True).to(device)
optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

model_path = os.path.join(save_path, "decoder_step_2000.pth")  # Replace with the correct saved model name

# Load model for testing
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    decoder.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

log_file_path = os.path.join(save_path, "training_loss.log")
with open(log_file_path, 'w') as log_file:
    # Training loop
    if TRAINING:
        print("***********************")
        print("Training decoder using " + str(steps) + " minibatches...")
        print("***********************")
        if FROZEN_TRAINING:
            set_requires_grad(decoder, test_layers)
        for step in range(steps):
            if ALL_ZEROS_CODEWORD_TRAINING:
                codewords = torch.zeros((n, batch_size))
            else:
                messages = torch.randint(0, 2, (k, batch_size)).float()
                codewords = (G @ messages) % 2

            BPSK_codewords = (0.5 - codewords) * 2

            # Initialize batch_data and create minibatch with codewords from multiple SNRs
            batch_data = torch.zeros_like(BPSK_codewords)
            SNRs = torch.arange(snr_lo, snr_hi + snr_step, snr_step)

            noise_per_snr = batch_size // len(SNRs)
            if TRAING_WISE_LAYER_NOISE:
                layers = test_layers
                layer_batch_sizes_weight = [int(batch_size * (FROZEN_TRAINING_alpha ** i)) for i in layers]
                layer_batch_sizes = [i / sum(layer_batch_sizes_weight) for i in layer_batch_sizes_weight]
                snr_values = [snr_lo + (snr_hi - snr_lo) * (1 - FROZEN_TRAINING_alpha ** i) for i in layers]
                batch_size_per_layer = [int(i * batch_size) for i in layer_batch_sizes]
                start_idx = 0
                snr_values = torch.tensor(snr_values)
                for i, snr in enumerate(snr_values):
                    sigma = torch.sqrt(1.0 / (2 * (k / n) * 10 ** (snr / 10)))
                    noise = sigma * torch.randn((n, batch_size_per_layer[i]))
                    batch_data[:, start_idx:start_idx + batch_size_per_layer[i]] = BPSK_codewords[:,
                                                                                   start_idx:start_idx +batch_size_per_layer[i]] + noise
                    start_idx += batch_size_per_layer[i]
                # print('snr:', snr_values)
                # print('batch_size_per_layer', batch_size_per_layer)
                remaining = batch_size - sum(batch_size_per_layer)
            else:
                start_idx = 0
                for i, snr in enumerate(SNRs):
                    sigma = torch.sqrt(1.0 / (2 * (k / n) * 10 ** (snr / 10)))
                    noise = sigma * torch.randn((n, noise_per_snr))
                    batch_data[:, start_idx:start_idx + noise_per_snr] = BPSK_codewords[:,
                                                                         start_idx:start_idx + noise_per_snr] + noise
                    start_idx += noise_per_snr
                remaining = batch_size % len(SNRs)

            # Handle remaining samples to ensure batch_size is fully populated
            if remaining > 0:
                for j in range(remaining):
                    idx = j % len(SNRs)  # Cycle through SNRs for remaining samples
                    sigma = torch.sqrt(1.0 / (2 * (k / n) * 10 ** (SNRs[idx] / 10)))
                    noise = sigma * torch.randn((n))
                    batch_data[:, start_idx + j] = BPSK_codewords[:, start_idx + j] + noise

            optimizer.zero_grad()

            # Compute loss and backpropagation
            loss, _ = decoder(batch_data.to(device), codewords.to(device), code)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Print loss at intervals
            if (step + 1) % 1 == 0:
                print(f"Step {step + 1}/{steps}, Loss: {loss.item()}")
                # print(decoder.B_cv)
                log_file.write(f"Step {step + 1}/{steps}, Loss: {loss.item()}\n")

            # Save model at intervals
            if (step + 1) % save_interval == 0:
                torch.save(decoder.state_dict(), os.path.join(save_path, f"decoder_step_2000_lst10_{step + 1}.pth"))
                print(f"Model saved at step {step + 1}")

# Testing loop
print("***********************")
print("Testing decoder...")
print("***********************")

BERs = []
FERs = []
SNRs = torch.arange(snr_lo, snr_hi + snr_step, snr_step)

for SNR in SNRs:
    sigma = torch.sqrt(1.0 / (2 * (k / n) * 10 ** (SNR / 10)))
    frame_count = 0
    bit_errors = 0
    frame_errors = 0
    FE = 0  # Frame Errors
    progress_bar = tqdm(total=max_frames, desc="Processing frames", ncols=100)

    while (FE < min_frame_errors or frame_count < min_frame) and frame_count < max_frames:
        progress_bar.update(batch_size)  # 每次循环更新一次进度条

        frame_count += batch_size

        # Generate codewords
        if ALL_ZEROS_CODEWORD_TESTING:
            codewords = torch.zeros((n, batch_size))
        else:
            messages = torch.randint(0, 2, (k, batch_size)).float()
            codewords = (G @ messages) % 2

        BPSK_codewords = (0.5 - codewords) * 2

        # Initialize batch_data and create minibatch with codewords from multiple SNRs
        batch_data = torch.zeros_like(BPSK_codewords)

        noise = sigma * np.random.randn(BPSK_codewords.shape[0], BPSK_codewords.shape[1])

        batch_data = BPSK_codewords + noise

        # Run the decoder
        _, soft_output = decoder(batch_data.to(device), codewords.to(device), code)
        recovered_codewords = (soft_output < 0).float()

        # Calculate errors
        errors = (codewords != recovered_codewords).float()  # Ensure errors is a tensor
        bit_errors += errors.sum().item()  # Sum of bit errors
        frame_errors += (errors.sum(dim=0) > 0).sum().item()  # Sum of frames with errors
        FE = frame_errors

    # Summarize results for this SNR
    bit_count = frame_count * n
    BER = bit_errors / bit_count
    FER = frame_errors / frame_count

    BERs.append(BER)
    FERs.append(FER)

    print(f"SNR: {SNR.item():.2f}, Frame count: {frame_count}, BER: {BER:.6f}, FER: {FER:.6f}")

# Print summary
print("Testing complete.")
print("BERs:", BERs)
print("FERs:", FERs)