import os
import argparse
import torch
import numpy as np
import csv
from scipy.spatial.distance import cosine
from Graph_embedding_model_initial_eye import (
    kan_embedding,
    graph_embedding2,
)  # Import your model
from utils import *

# Set the device for computation
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
device = torch.device("cuda:{}".format(device_ids[0]))
seed_everything(830)


# Function to compute cosine similarity between two vectors
def normalized_cosine_similarity(vec1, vec2):
    cos_sim = 1 - cosine(vec1, vec2)
    return (1 + cos_sim) / 2  # Normalize to [0, 1]


# Function to load test set filenames from CSV and exclude anchor subject
def load_test_filenames(test_csv_path, anchor_subject):
    test_filenames = []
    with open(test_csv_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            filename = row[0]
            # Exclude the anchor subject from the test set
            if filename != anchor_subject:
                test_filenames.append(filename)
    return test_filenames


# Function to load node input data for lh and rh (left and right hemispheres)
def load_node_input_data(data_path, hot_num):
    lh_data = []  # Left hemisphere data
    rh_data = []  # Right hemisphere data

    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_path, filename)
            node_data = torch.tensor(np.loadtxt(file_path), dtype=torch.float)
            if "_lh_" in filename:
                lh_data.append(node_data[0:hot_num, :])
            elif "_rh_" in filename:
                rh_data.append(node_data[0:hot_num, :])

    lh_data = torch.stack(lh_data) if lh_data else torch.tensor([])
    rh_data = torch.stack(rh_data) if rh_data else torch.tensor([])
    return lh_data, rh_data


# Function to load ROI data for lh and rh from adj_feature_matrix folder
def load_roi_data(roi_path, subject_id):
    lh_roi_file = os.path.join(roi_path, f"{subject_id}_3hinge_0_hop_feature_lh.txt")
    rh_roi_file = os.path.join(roi_path, f"{subject_id}_3hinge_0_hop_feature_rh.txt")

    lh_roi_data = (
        np.loadtxt(lh_roi_file) if os.path.exists(lh_roi_file) else np.array([])
    )
    rh_roi_data = (
        np.loadtxt(rh_roi_file) if os.path.exists(rh_roi_file) else np.array([])
    )

    return lh_roi_data, rh_roi_data


# Function to calculate ROI hit rate for top 1, top 3, and top 5 matches
def calculate_similarities(
    anchor_lh_data,
    anchor_rh_data,
    anchor_lh_roi,
    anchor_rh_roi,
    test_data_list,
    model,
    threshold,
):
    anchor_lh_data = anchor_lh_data.to(device)
    anchor_rh_data = anchor_rh_data.to(device)
    model.eval()

    (
        roi_hits_top1_lh,
        roi_hits_top3_lh,
        roi_hits_top5_lh,
        roi_hits_top7_lh,
        roi_hits_top10_lh,
    ) = (0, 0, 0, 0, 0)
    (
        roi_hits_top1_rh,
        roi_hits_top3_rh,
        roi_hits_top5_rh,
        roi_hits_top7_rh,
        roi_hits_top10_rh,
    ) = (0, 0, 0, 0, 0)
    total_nodes_lh, total_nodes_rh = 0, 0

    # Prepare lists for storing top 3 and top 5 cosine similarities for each anchor node
    top3_list = []
    top5_list = []
    top7_list = []
    top10_list = []

    with torch.no_grad():
        _, _, _, anchor_lh_embedding = model(anchor_lh_data)
        _, _, _, anchor_rh_embedding = model(anchor_rh_data)
        anchor_lh_embedding = anchor_lh_embedding.squeeze().cpu().numpy()
        anchor_rh_embedding = anchor_rh_embedding.squeeze().cpu().numpy()

    similarities = []
    count=0
    for subject, test_lh_data, test_rh_data, test_lh_roi, test_rh_roi in test_data_list:
        test_lh_data = test_lh_data.to(device)
        test_rh_data = test_rh_data.to(device)
        # print(test_lh_data.shape)
        # print(test_rh_data.shape)
        with torch.no_grad():
            _, _, _, test_lh_embedding = model(test_lh_data)
            _, _, _, test_rh_embedding = model(test_rh_data)
            test_lh_embedding = test_lh_embedding.squeeze().cpu().numpy()
            test_rh_embedding = test_rh_embedding.squeeze().cpu().numpy()
        count= count +test_rh_data.shape[0]+test_lh_data.shape[0]
        # print(test_lh_data.shape)
        # print(test_rh_data.shape)
        # print(subject)
        # for i in test_lh_data:
        #     with torch.no_grad():
        #         _, _, _, test_lh_embedding = model(test_lh_data[i])
        #         test_lh_embedding = test_lh_embedding.squeeze().cpu().numpy()
                
        # for i in test_rh_embedding:
        #     with torch.no_grad():
        #         _, _, _, test_rh_embedding = model(test_rh_data[i])
        #         test_rh_embedding = test_rh_embedding.squeeze().cpu().numpy()
        subject_lh_similarities = []
        subject_rh_similarities = []

        # Compare lh embeddings
        for i, anchor_node in enumerate(anchor_lh_embedding):
            # Collect all cosine similarities for this anchor node
            # print(anchor_node)
            sims_lh = [
                (j, normalized_cosine_similarity(anchor_node, test_node))
                for j, test_node in enumerate(test_lh_embedding)
            ]
            sims_lh.sort(
                key=lambda x: x[1], reverse=True
            )  # Sort by similarity in descending order

            top_matches_lh = [j for j, sim in sims_lh[:10]]  # Top 10 node indices
            top_sims_lh = [sim for j, sim in sims_lh[:10]]  # Top 10 cosine similarities

            # Apply threshold
            best_match_idx, best_sim = (
                sims_lh[0] if sims_lh[0][1] >= threshold else (-1, -1)
            )
            subject_lh_similarities.append((best_match_idx, best_sim))

            # Top 3 ROI check
            roi_hit_top1 = (
                np.array_equal(anchor_lh_roi[i], test_lh_roi[best_match_idx])
                if best_sim != -1
                else False
            )
            roi_hit_top3 = any(
                np.array_equal(anchor_lh_roi[i], test_lh_roi[j])
                for j in top_matches_lh[:3]
            )
            roi_hit_top5 = any(
                np.array_equal(anchor_lh_roi[i], test_lh_roi[j])
                for j in top_matches_lh[:5]
            )
            roi_hit_top7 = any(
                np.array_equal(anchor_lh_roi[i], test_lh_roi[j])
                for j in top_matches_lh[:7]
            )
            roi_hit_top10 = any(
                np.array_equal(anchor_lh_roi[i], test_lh_roi[j])
                for j in top_matches_lh[:10]
            )

            if roi_hit_top1:
                roi_hits_top1_lh += 1
            if roi_hit_top3:
                roi_hits_top3_lh += 1
            if roi_hit_top5:
                roi_hits_top5_lh += 1
            if roi_hit_top7:
                roi_hits_top7_lh += 1
            if roi_hit_top10:
                roi_hits_top10_lh += 1
            total_nodes_lh += 1

            # Save top 3 and top 5 matches for lh
            top3_list.append([i] + top_matches_lh[:3] + top_sims_lh[:3])
            top5_list.append([i] + top_matches_lh[:5] + top_sims_lh[:5])
            top7_list.append([i] + top_matches_lh[:7] + top_sims_lh[:7])
            top10_list.append([i] + top_matches_lh[:10] + top_sims_lh[:10])

        # Compare rh embeddings
        for i, anchor_node in enumerate(anchor_rh_embedding):
            sims_rh = [
                (j, normalized_cosine_similarity(anchor_node, test_node))
                for j, test_node in enumerate(test_rh_embedding)
            ]
            sims_rh.sort(key=lambda x: x[1], reverse=True)

            top_matches_rh = [j for j, sim in sims_rh[:10]]
            top_sims_rh = [sim for j, sim in sims_rh[:10]]

            best_match_idx, best_sim = (
                sims_rh[0] if sims_rh[0][1] >= threshold else (-1, -1)
            )
            subject_rh_similarities.append((best_match_idx, best_sim))

            roi_hit_top1 = (
                np.array_equal(anchor_rh_roi[i], test_rh_roi[best_match_idx])
                if best_sim != -1
                else False
            )
            roi_hit_top3 = any(
                np.array_equal(anchor_rh_roi[i], test_rh_roi[j])
                for j in top_matches_rh[:3]
            )
            roi_hit_top5 = any(
                np.array_equal(anchor_rh_roi[i], test_rh_roi[j])
                for j in top_matches_rh[:5]
            )
            roi_hit_top7 = any(
                np.array_equal(anchor_rh_roi[i], test_rh_roi[j])
                for j in top_matches_rh[:7]
            )
            roi_hit_top10 = any(
                np.array_equal(anchor_rh_roi[i], test_rh_roi[j])
                for j in top_matches_rh[:10]
            )

            if roi_hit_top1:
                roi_hits_top1_rh += 1
            if roi_hit_top3:
                roi_hits_top3_rh += 1
            if roi_hit_top5:
                roi_hits_top5_rh += 1
            if roi_hit_top7:
                roi_hits_top7_rh += 1
            if roi_hit_top10:
                roi_hits_top10_rh += 1
            total_nodes_rh += 1

            # Save top 3 and top 5 matches for rh
            top3_list.append([i] + top_matches_rh[:3] + top_sims_rh[:3])
            top5_list.append([i] + top_matches_rh[:5] + top_sims_rh[:5])
            top7_list.append([i] + top_matches_rh[:7] + top_sims_rh[:7])
            top10_list.append([i] + top_matches_rh[:10] + top_sims_rh[:10])

        similarities.append((subject, subject_lh_similarities, subject_rh_similarities))

    roi_hit_rate_top1_lh = (
        roi_hits_top1_lh / total_nodes_lh if total_nodes_lh > 0 else 0
    )
    roi_hit_rate_top3_lh = (
        roi_hits_top3_lh / total_nodes_lh if total_nodes_lh > 0 else 0
    )
    roi_hit_rate_top5_lh = (
        roi_hits_top5_lh / total_nodes_lh if total_nodes_lh > 0 else 0
    )
    roi_hit_rate_top7_lh = (
        roi_hits_top7_lh / total_nodes_lh if total_nodes_lh > 0 else 0
    )
    roi_hit_rate_top10_lh = (
        roi_hits_top10_lh / total_nodes_lh if total_nodes_lh > 0 else 0
    )

    roi_hit_rate_top1_rh = (
        roi_hits_top1_rh / total_nodes_rh if total_nodes_rh > 0 else 0
    )
    roi_hit_rate_top3_rh = (
        roi_hits_top3_rh / total_nodes_rh if total_nodes_rh > 0 else 0
    )
    roi_hit_rate_top5_rh = (
        roi_hits_top5_rh / total_nodes_rh if total_nodes_rh > 0 else 0
    )
    roi_hit_rate_top7_rh = (
        roi_hits_top7_rh / total_nodes_rh if total_nodes_rh > 0 else 0
    )
    roi_hit_rate_top10_rh = (
        roi_hits_top10_rh / total_nodes_rh if total_nodes_rh > 0 else 0
    )

    # Calculate combined ROI hit rates (LH + RH)
    total_hits_top1 = roi_hits_top1_lh + roi_hits_top1_rh
    total_hits_top3 = roi_hits_top3_lh + roi_hits_top3_rh
    total_hits_top5 = roi_hits_top5_lh + roi_hits_top5_rh
    total_hits_top7 = roi_hits_top7_lh + roi_hits_top7_rh
    total_hits_top10 = roi_hits_top10_lh + roi_hits_top10_rh
    total_nodes = total_nodes_lh + total_nodes_rh

    roi_hit_rate_top1_total = total_hits_top1 / total_nodes if total_nodes > 0 else 0
    roi_hit_rate_top3_total = total_hits_top3 / total_nodes if total_nodes > 0 else 0
    roi_hit_rate_top5_total = total_hits_top5 / total_nodes if total_nodes > 0 else 0
    roi_hit_rate_top7_total = total_hits_top7 / total_nodes if total_nodes > 0 else 0
    roi_hit_rate_top10_total = total_hits_top10 / total_nodes if total_nodes > 0 else 0

    print(count)
    # Output ROI hit rates
    print(f"ROI Hit Rates for LH:")
    print(f"Top 1: {roi_hit_rate_top1_lh:.4f}")
    print(f"Top 3: {roi_hit_rate_top3_lh:.4f}")
    print(f"Top 5: {roi_hit_rate_top5_lh:.4f}")
    print(f"Top 7: {roi_hit_rate_top7_lh:.4f}")
    print(f"Top 10: {roi_hit_rate_top10_lh:.4f}")

    print(f"ROI Hit Rates for RH:")
    print(f"Top 1: {roi_hit_rate_top1_rh:.4f}")
    print(f"Top 3: {roi_hit_rate_top3_rh:.4f}")
    print(f"Top 5: {roi_hit_rate_top5_rh:.4f}")
    print(f"Top 7: {roi_hit_rate_top7_rh:.4f}")
    print(f"Top 10: {roi_hit_rate_top10_rh:.4f}")

    print(f"Total ROI Hit Rates (LH + RH):")
    print(f"Top 1: {roi_hit_rate_top1_total:.4f}")
    print(f"Top 3: {roi_hit_rate_top3_total:.4f}")
    print(f"Top 5: {roi_hit_rate_top5_total:.4f}")
    print(f"Top 7: {roi_hit_rate_top7_total:.4f}")
    print(f"Top 10: {roi_hit_rate_top10_total:.4f}")

    return (
        similarities,
        (
            roi_hit_rate_top1_lh,
            roi_hit_rate_top3_lh,
            roi_hit_rate_top5_lh,
            roi_hit_rate_top7_lh,
            roi_hit_rate_top10_lh,
        ),
        (
            roi_hit_rate_top1_rh,
            roi_hit_rate_top3_rh,
            roi_hit_rate_top5_rh,
            roi_hit_rate_top7_rh,
            roi_hit_rate_top10_rh,
        ),
        top3_list,
        top5_list,
        top7_list,
        top10_list,
    )


# Function to save the results to a CSV file
def save_to_csv(
    output_path,
    similarities,
    roi_hit_rates_lh,
    roi_hit_rates_rh,
    top3_csv,
    top5_csv,
    top7_csv,
    top10_csv,
    top3_list,
    top5_list,
    top7_list,
    top10_list,
):
    with open(output_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "Anchor Node",
                "Test Subject",
                "Matched Node",
                "Cosine Similarity",
                "Hemisphere",
            ]
        )
        for subject, lh_similarities, rh_similarities in similarities:
            for anchor_idx, (matched_idx, similarity) in enumerate(lh_similarities):
                csvwriter.writerow([anchor_idx, subject, matched_idx, similarity, "lh"])
            for anchor_idx, (matched_idx, similarity) in enumerate(rh_similarities):
                csvwriter.writerow([anchor_idx, subject, matched_idx, similarity, "rh"])

        # Add ROI hit rate to the CSV file
        csvwriter.writerow([])
        csvwriter.writerow(["ROI Hit Rate LH (Top 1)", roi_hit_rates_lh[0]])
        csvwriter.writerow(["ROI Hit Rate LH (Top 3)", roi_hit_rates_lh[1]])
        csvwriter.writerow(["ROI Hit Rate LH (Top 5)", roi_hit_rates_lh[2]])
        csvwriter.writerow(["ROI Hit Rate LH (Top 7)", roi_hit_rates_lh[3]])
        csvwriter.writerow(["ROI Hit Rate LH (Top 10)", roi_hit_rates_lh[4]])
        csvwriter.writerow(["ROI Hit Rate RH (Top 1)", roi_hit_rates_rh[0]])
        csvwriter.writerow(["ROI Hit Rate RH (Top 3)", roi_hit_rates_rh[1]])
        csvwriter.writerow(["ROI Hit Rate RH (Top 5)", roi_hit_rates_rh[2]])
        csvwriter.writerow(["ROI Hit Rate RH (Top 7)", roi_hit_rates_rh[3]])
        csvwriter.writerow(["ROI Hit Rate RH (Top 10)", roi_hit_rates_rh[4]])

    # Save top 3 matches
    with open(top3_csv, "w", newline="") as top3_file:
        top3_writer = csv.writer(top3_file)
        top3_writer.writerow(
            [
                "Anchor Node",
                "Top 1 Node",
                "Top 2 Node",
                "Top 3 Node",
                "Top 1 Sim",
                "Top 2 Sim",
                "Top 3 Sim",
            ]
        )
        top3_writer.writerows(top3_list)

    # Save top 5 matches
    with open(top5_csv, "w", newline="") as top5_file:
        top5_writer = csv.writer(top5_file)
        top5_writer.writerow(
            [
                "Anchor Node",
                "Top 1 Node",
                "Top 2 Node",
                "Top 3 Node",
                "Top 4 Node",
                "Top 5 Node",
                "Top 1 Sim",
                "Top 2 Sim",
                "Top 3 Sim",
                "Top 4 Sim",
                "Top 5 Sim",
            ]
        )
        top5_writer.writerows(top5_list)
    # Save top 7 matches
    with open(top7_csv, "w", newline="") as top7_file:
        top7_writer = csv.writer(top7_file)
        top7_writer.writerow(
            [
                "Anchor Node",
                "Top 1 Node",
                "Top 2 Node",
                "Top 3 Node",
                "Top 4 Node",
                "Top 5 Node",
                "Top 6 Node",
                "Top 7 Node",
                "Top 1 Sim",
                "Top 2 Sim",
                "Top 3 Sim",
                "Top 4 Sim",
                "Top 5 Sim",
                "Top 6 Sim",
                "Top 7 Sim",
            ]
        )
        top7_writer.writerows(top7_list)
    # Save top 10 matches
    with open(top10_csv, "w", newline="") as top10_file:
        top10_writer = csv.writer(top10_file)
        top10_writer.writerow(
            [
                "Anchor Node",
                "Top 1 Node",
                "Top 2 Node",
                "Top 3 Node",
                "Top 4 Node",
                "Top 5 Node",
                "Top 6 Node",
                "Top 7 Node",
                "Top 8 Node",
                "Top 9 Node",
                "Top 10 Node",
                "Top 1 Sim",
                "Top 2 Sim",
                "Top 3 Sim",
                "Top 4 Sim",
                "Top 5 Sim",
                "Top 6 Sim",
                "Top 7 Sim",
                "Top 8 Sim",
                "Top 9 Sim",
                "Top 10 Sim",
            ]
        )
        top10_writer.writerows(top10_list)


# Main function to process the testing set and compute similarities
def main(args):
    # Load the pre-trained model
    model = graph_embedding2(args.embedding_num, args.embedding_dim, args.hot_num)
    # model = kan_embedding(args.embedding_num, args.embedding_dim, args.hot_num)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # Load anchor subject's node input data and ROI data
    # anchor_path = os.path.join(args.test_root, args.anchor_subject, "Graph_embedding_data/node_input_data")
    anchor_path = os.path.join(
        args.test_root, args.anchor_subject, "Graph_embedding_data/sse_node_input_data"
    )
    # anchor_path = os.path.join(
    #     args.test_root, args.anchor_subject, "Graph_embedding_data/bsse_node_input_data"
    # )
    anchor_lh_data, anchor_rh_data = load_node_input_data(anchor_path, args.hot_num)

    anchor_roi_path = os.path.join(
        args.test_root, args.anchor_subject, "Graph_embedding_data/adj_feature_matrix"
    )
    anchor_lh_roi, anchor_rh_roi = load_roi_data(anchor_roi_path, args.anchor_subject)

    # Load test subjects' node input data and ROI data from CSV, excluding the anchor subject
    test_filenames = load_test_filenames(args.test_csv, args.anchor_subject)
    test_data_list = []
    for subject_folder in test_filenames:
        # test_data_path = os.path.join(args.test_root, subject_folder, "Graph_embedding_data/node_input_data")
        test_data_path = os.path.join(
            args.test_root, subject_folder, "Graph_embedding_data/sse_node_input_data"
        )
        # test_data_path = os.path.join(
        #     args.test_root, subject_folder, "Graph_embedding_data/bsse_node_input_data"
        # )
        lh_data, rh_data = load_node_input_data(test_data_path, args.hot_num)

        roi_path = os.path.join(
            args.test_root, subject_folder, "Graph_embedding_data/adj_feature_matrix"
        )
        lh_roi, rh_roi = load_roi_data(roi_path, subject_folder)

        test_data_list.append((subject_folder, lh_data, rh_data, lh_roi, rh_roi))

    # Calculate similarities and ROI hit rates
    (
        similarities,
        roi_hit_rates_lh,
        roi_hit_rates_rh,
        top3_list,
        top5_list,
        top7_list,
        top10_list,
    ) = calculate_similarities(
        anchor_lh_data,
        anchor_rh_data,
        anchor_lh_roi,
        anchor_rh_roi,
        test_data_list,
        model,
        args.threshold,
    )

    # Save results to CSV
    save_to_csv(
        args.output_path,
        similarities,
        roi_hit_rates_lh,
        roi_hit_rates_rh,
        args.top3_csv,
        args.top5_csv,
        args.top7_csv,
        args.top10_csv,
        top3_list,
        top5_list,
        top7_list,
        top10_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate node correspondences based on cosine similarity."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../hop_2/Graph_embedding_model-6.ckpt",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default="/media/minheng/hdd_3/HCP_cc_0819/HCP_new/",
        help="Root directory of the testing set",
    )
    parser.add_argument(
        "--anchor_subject",
        type=str,
        default="100206",
        help="Folder name of the anchor subject",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="./test_files.csv",
        help="Path to the CSV file containing test set filenames",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./c2v_max_correspondence.csv",
        help="Path to save the main CSV file",
    )
    parser.add_argument(
        "--top3_csv",
        type=str,
        default="./c2v_top3_correspondence.csv",
        help="Path to save the top 3 matches CSV file",
    )
    parser.add_argument(
        "--top5_csv",
        type=str,
        default="./c2v_top5_correspondence.csv",
        help="Path to save the top 5 matches CSV file",
    )
    parser.add_argument(
        "--top7_csv",
        type=str,
        default="./c2v_top7_correspondence.csv",
        help="Path to save the top 7 matches CSV file",
    )
    parser.add_argument(
        "--top10_csv",
        type=str,
        default="./c2v_top10_correspondence.csv",
        help="Path to save the top 10 matches CSV file",
    )
    parser.add_argument(
        "--embedding_num",
        type=int,
        default=75,
        help="Number of embeddings for the model",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Dimension of embeddings for the model",
    )
    parser.add_argument(
        "--hot_num", type=int, default=3, help="Number of hot dimensions"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for considering a match",
    )

    args = parser.parse_args()
    main(args)
