import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="plot simple performance plot")
    parser.add_argument('--test_file_path', type=str, required=True, help = "Model name ")
    args = parser.parse_args()

    return args

def plot_mbb_classifier_score(mbb, labels, classifier_scores, analysis_nn_scores, save_path):
    # Create bins in mbb
    mbb_bins = np.linspace(60, 180, 13)  # e.g., 12 bins from 60 to 180 GeV
    bin_indices = np.digitize(mbb, mbb_bins)

    # Group classifier score by mbb bins
    binned_scores = [classifier_scores[bin_indices == i] for i in range(1, len(mbb_bins))]
    binned_labels = [labels[bin_indices == i] for i in range(1, len(mbb_bins))]

    binned_scores_nn = [analysis_nn_scores[bin_indices == i] for i in range(1, len(mbb_bins))]

    # Plot average classifier score per mbb bin (decorrelation check)
    avg_scores = [np.mean(scores) if len(scores) > 0 else np.nan for scores in binned_scores]
    bin_centers = 0.5 * (mbb_bins[:-1] + mbb_bins[1:])

    avg_nn_scores = [np.mean(scores) if len(scores) > 0 else np.nan for scores in binned_scores_nn]

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, avg_scores, marker='o', label='Avg classifier score')
    plt.plot(bin_centers, avg_nn_scores, marker='o', label='Avg analysis score')
    plt.xlabel("m_bb [GeV]")
    plt.ylabel("Avg classifier output")
    plt.title("Sanity Check: Classifier Score vs m_bb (should be flat)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path+"_mbb_vs_classifier_scpre")
    plt.legend()
    plt.show()

def adversial_score_vs_mbb( mbb, adv_pred, save_path ):
    plt.figure(figsize=(8, 6))
    sns.histplot(x=mbb, hue=adv_pred, bins=20, palette="viridis", multiple="stack")
    plt.xlabel("m_bb [GeV]")
    plt.ylabel("Events per bin")
    plt.title("Adversary Prediction vs m_bb")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path+"_mbb_vs_adverserial_score")
    plt.show()

def main(filepath):
    with h5py.File(filepath, 'r') as file:
        events = file["events"]
        class_label = events["label"]
        mbb = events["mBB"] / 1000  # Convert MeV to GeV for readability
        classifier_scores = events["classifier_scores"]
        adv_scores = events["adverserial_scores"]
        analysis_nn_scores = events["nn_Sideband_Data_newgrlbv_v1_lambda100"]
        plot_mbb_classifier_score(mbb, class_label, classifier_scores, analysis_nn_scores, "plots/Test/")
        adversial_score_vs_mbb(mbb, adv_scores, "plots/Test/")

if __name__ == "__main__":
    args = parse_args()
    main(args.test_file_path)


