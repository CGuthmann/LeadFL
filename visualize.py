import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logging/Final/federator.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['round_id'], df['test_accuracy'], label='Test Accuracy')
plt.plot(df['round_id'], df['backdoor_accuracy'], label='Backdoor Attack Success Rate', linestyle='--')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Model Accuracy vs Backdoor Attack Success')
plt.savefig('backdoor_tracking_leadfl_org_80_noniid.png')
plt.show()