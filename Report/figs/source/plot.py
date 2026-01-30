import pandas as pd
import matplotlib.pyplot as plt


def periodic_ma():
    # =========================
    # 1. Load CSV data
    # =========================
    csv_path = "mnist_ma_periodic.csv"  # <-- change if needed
    df = pd.read_csv(csv_path)

    # =========================
    # 2. Column configuration
    # =========================
    round_col = "round"

    accuracy_columns = {
        0.2: "acc_alpha_0_2",
        0.4: "acc_alpha_0_4",
        0.6: "acc_alpha_0_6",
    }

    malicious_col = "malicious_clients"

    # =========================
    # 3. Create figure and axes
    # =========================
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # =========================
    # 4. Plot backdoor accuracy
    # =========================
    styles = {
        0.2: dict(color="orange", linestyle="-", linewidth=2),
        0.4: dict(color="red", linestyle="--", linewidth=2),
        0.6: dict(color="purple", linestyle="-.", linewidth=2),
    }

    for alpha, col in accuracy_columns.items():
        ax1.plot(
            df[round_col]+1,
            df[col],
            label=f"alpha={alpha}",
            **styles[alpha]
        )

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Maintask Accuracy")
    
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylim(75, 90)
    ax1.margins(y=0)
    # =========================
    # 5. Secondary axis:
    #    malicious clients
    # =========================
    ax2 = ax1.twinx()

    ax2.plot(
        df[round_col]+1,
        df[malicious_col],
        color="black",
        linestyle=":",
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax2.set_ylabel("Number of Malicious Clients", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    xticks = ax1.get_xticks()

    # Add x = 1 if not already present
    if 1 not in xticks:
        ax1.set_xticks(sorted(list(xticks) + [1]))
        
    ax1.set_xlim(1, 40)
    # =========================
    # 6. Legend and title
    # =========================
    ax1.legend(loc="upper right")
    #plt.title("Effect of Regularization Rate on Maintask Accuracy")

    plt.tight_layout()
    plt.show()

def periodic_ma_defenses():
    # =========================
    # 1. Load CSV data
    # =========================
    csv_path = "mnist_ma_periodic.csv"  # <-- change if needed
    df = pd.read_csv(csv_path)

    # =========================
    # 2. Column configuration
    # =========================
    round_col = "round"

    accuracy_columns = {
        "Our LeadFL"            : "acc_alpha_0_2",
        "Original LeadFL"       :      "acc_alpha_0_4",
        "FL-WBC"                : "WBC",
        "LDP"                   :  "LDP",
        "None"            : "None",
    }

    malicious_col = "malicious_clients"

    # =========================
    # 3. Create figure and axes
    # =========================
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # =========================
    # 4. Plot backdoor accuracy
    # =========================
    styles = {
        "Our LeadFL": dict(color="orange", linestyle="solid", linewidth=2),
        "Original LeadFL": dict(color="red", linestyle="dotted", linewidth=2),
        "LeadFL": dict(color="purple", linestyle="dashed", linewidth=2),
        "FL-WBC": dict(color="blue", linestyle=(0, (1, 1))),
        "LDP": dict(color="darkgreen", linestyle="loosely dashed", linewidth=2), 
        "None": dict(color="teal", linestyle="dashdotdotted", linewidth=2),        
                        
        "Our LeadFL" : dict(color="orange", linestyle="-"),
        "Original LeadFL" : dict(color="red", linestyle="--"),
        "LeadFL": dict(color="purple", linestyle="-."),
        "FL-WBC" : dict(color="blue", linestyle=(0, (1, 1))),
        "LDP": dict(color="green", linestyle=(0, (6, 2))),
        "None": dict(color="brown", linestyle=(0, (4, 2, 1, 2))),
                          
    }

    for name, col in accuracy_columns.items():
        ax1.plot(
            df[round_col]+1,
            df[col],
            label=f"{name}",
            **styles[name]
        )

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Maintask Accuracy")
    ax1.set_ylim(0, 100)
    
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylim(75, 90)
    ax1.margins(y=0)
    # =========================
    # 5. Secondary axis:
    #    malicious clients
    # =========================
    ax2 = ax1.twinx()

    ax2.plot(
        df[round_col]+1,
        df[malicious_col],
        color="black",
        linestyle=":",
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax2.set_ylabel("Number of Malicious Clients", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    xticks = ax1.get_xticks()

    # Add x = 1 if not already present
    if 1 not in xticks:
        ax1.set_xticks(sorted(list(xticks) + [1]))
        
    ax1.set_xlim(1, 40)
    # =========================
    # 6. Legend and title
    # =========================
    ax1.legend(loc="upper right")
    #plt.title("Effect of Regularization Rate on Maintask Accuracy")

    plt.tight_layout()
    plt.show()

def random_ma_defenses():
    # =========================
    # 1. Load CSV data
    # =========================
    csv_path = "mnist_ma_random.csv"  # <-- change if needed
    df = pd.read_csv(csv_path)

    # =========================
    # 2. Column configuration
    # =========================
    round_col = "round"

    accuracy_columns = {
        "Our LeadFL"            : "LeadFL",
        "Original LeadFL"       :      "LeadFL",
        "FL-WBC"                : "WBC",
        "LDP"                   :  "LDP",
        "None"            : "None",
    }

    malicious_col = "malicious_clients"

    # =========================
    # 3. Create figure and axes
    # =========================
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # =========================
    # 4. Plot backdoor accuracy
    # =========================
    styles = { 
                        
        "Our LeadFL" : dict(color="orange", linestyle="-"),
        "Original LeadFL" : dict(color="red", linestyle="--"),
        "LeadFL": dict(color="purple", linestyle="-."),
        "FL-WBC" : dict(color="blue", linestyle=(0, (1, 1))),
        "LDP": dict(color="green", linestyle=(0, (6, 2))),
        "None": dict(color="brown", linestyle=(0, (4, 2, 1, 2))),
                          
    }

    for name, col in accuracy_columns.items():
        ax1.plot(
            df[round_col]+1,
            df[col],
            label=f"{name}",
            **styles[name]
        )

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Maintask Accuracy")
    
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylim(75, 90)
    ax1.margins(y=0)
    # =========================
    # 5. Secondary axis:
    #    malicious clients
    # =========================
    ax2 = ax1.twinx()

    ax2.plot(
        df[round_col]+1,
        df[malicious_col],
        color="black",
        linestyle=":",
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax2.set_ylabel("Number of Malicious Clients", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    xticks = ax1.get_xticks()

    # Add x = 1 if not already present
    if 1 not in xticks:
        ax1.set_xticks(sorted(list(xticks) + [1]))
        
    ax1.set_xlim(1, 40)
    # =========================
    # 6. Legend and title
    # =========================
    ax1.legend(loc="upper right")
    #plt.title("Effect of Regularization Rate on Maintask Accuracy")

    plt.tight_layout()
    plt.show()

def periodic_ba():
    # =========================
    # 1. Load CSV data
    # =========================
    csv_path = "mnist_ba_periodic.csv"  
    df = pd.read_csv(csv_path)

    # =========================
    # 2. Column configuration
    # =========================
    round_col = "round"

    accuracy_columns = {
        0.2: "acc_alpha_0_2",
        0.4: "acc_alpha_0_4",
        0.6: "acc_alpha_0_6",
    }

    malicious_col = "malicious_clients"

    # =========================
    # 3. Create figure and axes
    # =========================
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # =========================
    # 4. Plot backdoor accuracy
    # =========================
    styles = {
        0.2: dict(color="orange", linestyle="-", linewidth=2),
        0.4: dict(color="red", linestyle="--", linewidth=2),
        0.6: dict(color="purple", linestyle="-.", linewidth=2),
    }

    for alpha, col in accuracy_columns.items():
        ax1.plot(
            df[round_col],
            df[col],
            label=f"alpha={alpha}",
            **styles[alpha]
        )

    ax1.set_xlabel("Communication round")
    ax1.set_xlim(1,40)
    ax1.set_ylabel("Backdoor Accuracy")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylim(0, 102)
    ax1.margins(y=0)
    
    # =========================
    # 5. Secondary axis:
    #    malicious clients
    # =========================
    ax2 = ax1.twinx()

    ax2.plot(
        df[round_col],
        df[malicious_col],
        color="black",
        linestyle=":",
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax2.set_ylabel("Number of Malicious Clients", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    xticks = ax1.get_xticks()

    # Add x = 1 if not already present
    if 1 not in xticks:
        ax1.set_xticks(sorted(list(xticks) + [1]))
        
    ax1.set_xlim(1, 40)
    # =========================
    # 6. Legend and title
    # =========================
    ax1.legend(loc="upper right")
    #plt.title("Effect of Regularization Rate on Backdoor Accuracy")

    plt.tight_layout()
    plt.show()

def periodic_ba_defenses():
    # =========================
    # 1. Load CSV data
    # =========================
    csv_path = "mnist_ba_periodic.csv"  # <-- change if needed
    df = pd.read_csv(csv_path)

    # =========================
    # 2. Column configuration
    # =========================
    round_col = "round"

    accuracy_columns = {
        "Our LeadFL"            : "acc_alpha_0_2",
        "Original LeadFL"       :      "acc_alpha_0_4",
        "FL-WBC"                : "WBC",
        "LDP"                   :  "LDP",
        "None"            : "None",
    }

    malicious_col = "malicious_clients"

    # =========================
    # 3. Create figure and axes
    # =========================
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # =========================
    # 4. Plot backdoor accuracy
    # =========================
    styles = { 
                        
        "Our LeadFL" : dict(color="orange", linestyle="-"),
        "Original LeadFL" : dict(color="red", linestyle="--"),
        "LeadFL": dict(color="purple", linestyle="-."),
        "FL-WBC" : dict(color="blue", linestyle=(0, (1, 1))),
        "LDP": dict(color="green", linestyle=(0, (6, 2))),
        "None": dict(color="brown", linestyle=(0, (4, 2, 1, 2))),
                          
    }

    for name, col in accuracy_columns.items():
        ax1.plot(
            df[round_col]+1,
            df[col],
            label=f"{name}",
            **styles[name]
        )

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Backdoor Accuracy")
    
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylim(0, 102)
    ax1.margins(y=0)
    # =========================
    # 5. Secondary axis:
    #    malicious clients
    # =========================
    ax2 = ax1.twinx()

    ax2.plot(
        df[round_col]+1,
        df[malicious_col],
        color="black",
        linestyle=":",
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax2.set_ylabel("Number of Malicious Clients", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    xticks = ax1.get_xticks()

    # Add x = 1 if not already present
    if 1 not in xticks:
        ax1.set_xticks(sorted(list(xticks) + [1]))
        
    ax1.set_xlim(1, 40)
    # =========================
    # 6. Legend and title
    # =========================
    ax1.legend(loc="upper right")
    #plt.title("Effect of Regularization Rate on Maintask Accuracy")

    plt.tight_layout()
    plt.show()

def random_ba_defenses():
    # =========================
    # 1. Load CSV data
    # =========================
    csv_path = "mnist_ba_random.csv"  # <-- change if needed
    df = pd.read_csv(csv_path)

    # =========================
    # 2. Column configuration
    # =========================
    round_col = "round"

    accuracy_columns = {
        "Our LeadFL"            : "LeadFL",
        "Original LeadFL"       :      "LeadFL",
        "FL-WBC"                : "WBC",
        "LDP"                   :  "LDP",
        "None"            : "None",
    }

    malicious_col = "malicious_clients"

    # =========================
    # 3. Create figure and axes
    # =========================
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # =========================
    # 4. Plot backdoor accuracy
    # =========================
    styles = { 
                        
        "Our LeadFL" : dict(color="orange", linestyle="-"),
        "Original LeadFL" : dict(color="red", linestyle="--"),
        "LeadFL": dict(color="purple", linestyle="-."),
        "FL-WBC" : dict(color="blue", linestyle=(0, (1, 1))),
        "LDP": dict(color="green", linestyle=(0, (6, 2))),
        "None": dict(color="brown", linestyle=(0, (4, 2, 1, 2))),
                          
    }

    for name, col in accuracy_columns.items():
        ax1.plot(
            df[round_col]+1,
            df[col],
            label=f"{name}",
            **styles[name]
        )

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Backdoor Accuracy")
    
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.set_ylim(0, 102)
    ax1.margins(y=0)
    # =========================
    # 5. Secondary axis:
    #    malicious clients
    # =========================
    ax2 = ax1.twinx()

    ax2.plot(
        df[round_col]+1,
        df[malicious_col],
        color="black",
        linestyle=":",
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    ax2.set_ylabel("Number of Malicious Clients", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    xticks = ax1.get_xticks()

    # Add x = 1 if not already present
    if 1 not in xticks:
        ax1.set_xticks(sorted(list(xticks) + [1]))
        
    ax1.set_xlim(1, 40)
    # =========================
    # 6. Legend and title
    # =========================
    ax1.legend(loc="upper right")
    #plt.title("Effect of Regularization Rate on Maintask Accuracy")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    periodic_ba_defenses()
    random_ba_defenses()