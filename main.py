import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataloader import create_federated_datasets
from model import BiLSTMTCNGAN
from regularizers import JacobianRegularizer, FedProxRegularizer, LeCamDivergence, L2Regularizer
# =========================================================
# OPTIONAL SYSTEM, MODE COLLAPSE PROFILING and BASELINES (DISABLED when needed)
# =========================================================
# import time
# import psutil
# import os
# from mode_collapse import compute_mode_collapse
# from baseline import FedTSRGNet, FedTrustNet, ADGAN, FedGANIDS
# =========================================================


def calculate_attack_detection_rate(precision, recall, lssd):
    """
    ADS = (Precision + Recall + (1 - SSD)) / 3
    """
    return (precision + recall + (1 - lssd)) / 3
# =========================================================
# OPTIONAL COMMUNICATION OVERHEAD FUNCTION (DISABLED)
# =========================================================
# def calculate_comm_overhead(model, num_clients, num_rounds):
#     total_params = sum(p.numel() for p in model.parameters())
#     total_bytes = total_params * 4 * 2 * num_clients * num_rounds
#     return total_bytes / (1024 ** 2)  # MB
# =========================================================


def train_federated_model(
    file_path,
    num_epochs=1,
    lr=0.001,
    batch_size=32,
    iid=True,       # False for non-IID
    labelled=True,  # False for Unlabelled
    #ablation_mode="FedGAD"   # FedAvg | FedProx | LeCam | FedGAD
):
    # =====================================================
    # OPTIONAL RUNTIME & MEMORY PROFILING START (DISABLED)
    # =====================================================
    # start_time = time.time()
    # process = psutil.Process(os.getpid())
    # start_mem = process.memory_info().rss / (1024 ** 2)
    # =====================================================
    #  Disable this lines when need to test with the baselines
    #========================================================
    # if MODEL_NAME == "FedGAD":
    # global_model = BiLSTMTCNGAN(input_dim).to(device)
    # elif MODEL_NAME == "FedTSRGNet":
    # global_model = FedTSRGNet(input_dim).to(device)
    # elif MODEL_NAME == "FedTrust":
    # global_model = FedTrustNet(input_dim).to(device)
    # elif MODEL_NAME == "ADGAN":
    # global_model = ADGAN(input_dim).to(device)
    # elif MODEL_NAME == "FedGANIDS":
    # global_model = FedGANIDS(input_dim).to(device)

    # ------------------ Load Data ------------------
    train_dataset, test_dataset = create_federated_datasets(
        file_path=file_path,
        iid=iid,
        labelled=labelled
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    input_size = train_dataset[0][0].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ Models ------------------
    model = BiLSTMTCNGAN(
        input_size=input_size,
        hidden_size=256,
        num_classes=2
    ).to(device)

    global_model = BiLSTMTCNGAN(
        input_size=input_size,
        hidden_size=256,
        num_classes=2
    ).to(device)

    global_model.load_state_dict(model.state_dict())

    # if MODEL_NAME == "FedGAD":
    # jacobian = JacobianRegularizer(
    # lambda_base=LAMBDA_BASE,
    # alpha=ALPHA,
    # adaptive=(ABLATION_MODE == "FedGAD-Full")
    # )
    # l2reg = L2Regularizer(L2_WEIGHT)
    # lecam_reg = LeCamDivergence(LECAM_WEIGHT)

    # ------------------ Optimizer & Loss ------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------ Regularizers ------------------
    jacobian_reg = JacobianRegularizer(weight=0.01)
    fedprox_reg = FedProxRegularizer(mu=0.01)
    lecam_reg = LeCamDivergence(weight=0.01)

    print(
        f"Epochs={num_epochs}, LR={lr}, Batch={batch_size}, "
        f"IID={iid}, Labeled={labelled}, Mode={ablation_mode}"
    )

    # =================== Training ===================
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_reg_loss = 0.0
        total_disc_loss = 0.0
        total_gen_loss = 0.0
        total_ssd_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # -------- Classification --------
            outputs = model(features.unsqueeze(1))
            cls_loss = criterion(outputs, labels)

            # -------- GAN Components --------
            z = torch.randn(features.size(0), 100).to(device)
            fake_data = model.generate_synthetic_data(z)
            real_data = features

            disc_real = model.discriminate(real_data)
            disc_fake = model.discriminate(fake_data.detach())

            d_loss = -torch.mean(
                torch.log(disc_real + 1e-8) +
                torch.log(1 - disc_fake + 1e-8)
            )

            g_loss = -torch.mean(
                torch.log(model.discriminate(fake_data) + 1e-8)
            )

            ssd_loss = nn.MSELoss()(
                fake_data, torch.zeros_like(fake_data)
            )

            # -------- Regularization --------
            reg_loss = 0.0
            if ablation_mode == "FedProx":
                reg_loss = fedprox_reg.compute(
                    list(model.parameters()),
                    list(global_model.parameters())
                )
            elif ablation_mode == "LeCam":
                reg_loss = lecam_reg.compute(real_data, fake_data)
            elif ablation_mode == "FedGAD":
                reg_loss = jacobian_reg.compute(
                    model.discriminate, real_data
                )
            elif ablation_mode == "FedAvg":
                reg_loss = 0.0

            loss = cls_loss + reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reg_loss += reg_loss.item() if torch.is_tensor(reg_loss) else 0.0
            total_disc_loss += d_loss.item()
            total_gen_loss += g_loss.item()
            total_ssd_loss += ssd_loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Loss={total_loss:.4f} | "
            f"Reg={total_reg_loss:.4f} | "
            f"D={total_disc_loss:.4f} | "
            f"G={total_gen_loss:.4f} | "
            f"SSD={total_ssd_loss:.4f}"
        )

    # =================== Evaluation ===================
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features.unsqueeze(1))
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    z = torch.randn(batch_size, 100).to(device)
    synthetic_data = model.generate_synthetic_data(z)
    ssd = nn.MSELoss()(
        synthetic_data, torch.zeros_like(synthetic_data)
    )

    ads = calculate_attack_detection_rate(
        precision, recall, ssd.item()
    )

    print(
        f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
        f"Recall={recall:.4f}, F1={f1:.4f}, ADS={ads:.4f}"
    )
# =====================================================
    # OPTIONAL MODE COLLAPSE & SYSTEM METRICS (DISABLED when needed)
    # =====================================================
    # missing_modes, mode_coverage = compute_mode_collapse(
    #     global_model,
    #     num_modes=8,
    #     latent_dim=100,
    #     device=device
    # )

    # end_time = time.time()
    # runtime = end_time - start_time
    # end_mem = process.memory_info().rss / (1024 ** 2)
    # peak_mem = max(start_mem, end_mem)

    # comm_overhead = calculate_comm_overhead(
    #     global_model,
    #     num_clients=100,
    #     num_rounds=num_epochs
    # )

    # with open("system_metrics.txt", "a") as f:
    #     f.write(
    #         f"{ablation_mode},"
    #         f"Runtime={runtime:.2f}s,"
    #         f"Memory={peak_mem:.2f}MB,"
    #         f"Comm={comm_overhead:.2f}MB,"
    #         f"MissingModes={missing_modes},"
    #         f"Coverage={mode_coverage:.2f}%\n"
    # )
    # =====================================================

    # =================== SAVE RESULTS ===================
    result_file = f"results_{ablation_mode}.txt"
    with open(result_file, "w") as f:
        f.write(
            f"Mode={ablation_mode}, "
            f"IID={iid}, Labeled={labelled}, "
            f"Acc={accuracy:.4f}, "
            f"Prec={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"ADS={ads:.4f}, "
            #f"GenLoss={total_gen_loss:.4f}, "
            #f"DiscLoss={total_disc_loss:.4f}, "
            #f"RegLoss={total_reg_loss:.4f}, "
            #f"SSDLoss={total_ssd_loss:.4f}, "
            #f"TotalLoss={total_loss:.4f}\n"
        )

    return model


# ====================== MAIN ======================
if __name__ == "__main__":
    file_path = "ToN_IoT.csv"     # Change it to CSE_CIC_IDS

    print("Labelled IID")
    train_federated_model(file_path, iid=True, labelled=True)

    #print("\nLabelled non-IID")
    #train_federated(file, iid=False, labelled=True)

    #print("\nUnlabelled IID")
    #train_federated(file, iid=True, labelled=False)

    #print("\nUnlabelled non-IID")
    #train_federated(file, iid=False, labelled=False)
