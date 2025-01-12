import os
import shutil
import xlsxwriter
import xlrd
import numpy as np
from collections import defaultdict
import nibabel.freesurfer.io as io
import pdb
import matplotlib.pyplot as plt
from dtw import dtw
from tqdm import tqdm
import csv
import pandas as pd
def create_or_clear_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory
        os.makedirs(directory)
        print(f"Directory {directory} has been created.")
    else:
        # If it exists, clear the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print(f"Directory {directory} has been cleared.")

def generate_node_feature(
    one_hot_feature, hop_1_feature, hop_2_feature, hop_3_feature, outdir, prefix
):
    for node in range(one_hot_feature.shape[0]):
        node_feature = np.zeros(shape=(4, one_hot_feature.shape[1]))
        node_feature[0, :] = one_hot_feature[node, :]
        node_feature[1, :] = hop_1_feature[node, :]
        node_feature[2, :] = hop_2_feature[node, :]
        node_feature[3, :] = hop_3_feature[node, :]
        np.savetxt(
            outdir + "/" + prefix + "_" + str(node) + ".txt", node_feature, fmt="%.1f"
        )


def d(a, b):
    # Calculate the maximum and minimum of a and b
    maximum = max(a, b)
    minimum = min(a, b)

    # Compute the function d(a, b)
    if a==0 or b==0:
        result = maximum
    else:
        result = (maximum / minimum) - 1

    return result


def calculate_degree_matrix(A):
    # Calculate the degree of each node by summing up each row of the adjacency matrix

    degrees = np.sum(A, axis=1)
    # Create a diagonal matrix where the diagonal elements are the node degrees
    D = np.diag(degrees)
    return degrees, D


def get_k_hop_binary_matrix(A, k):
    # Compute the k-hop matrix by raising the adjacency matrix to the power of k
    k_hop_matrix = np.linalg.matrix_power(A, k)

    # Convert all non-zero elements to 1
    binary_matrix = (k_hop_matrix > 0).astype(int)

    return binary_matrix


def get_k_hop_matrix(A, k):
    # Compute the k-hop matrix by raising the adjacency matrix to the power of k
    return np.linalg.matrix_power(A, k)

def detect_error(A, sphere, subject):
    '''
    Detecting isolated nodes on GryalNet
    '''
    # Calculate degree of each node
    degrees = np.sum(A, axis=1)
    
    # Find the row id of the node whose degree value is 1
    degree_one_nodes = np.where(degrees == 1)[0]

    
    return degree_one_nodes
    


def calculate_structural_similarity(adj_1_hop, need_identity=True):
    '''
    input: the adjacent matrix A claimed by Lu (which is actually A+I)
    output: the i-hop structural similarity matrix (same size as A), i-hop binary adjacent matrix and the identity matrix I(optional)
    '''
    # calculate the real adjacent matrix
    I = np.eye(adj_1_hop.shape[0])
    A = adj_1_hop - I
    degrees, D = calculate_degree_matrix(A)

    A_2 = get_k_hop_matrix(A, 2)
    A_3 = get_k_hop_matrix(A, 3)

    s_0 = np.ones_like(A)
    s_1 = np.ones_like(A)
    s_2 = np.ones_like(A)
    s_3 = np.ones_like(A)
    binary_A_2 = get_k_hop_binary_matrix(A, 2)
    binary_A_3 = get_k_hop_binary_matrix(A, 3)
    binary_A_4 = get_k_hop_binary_matrix(A, 4)
    for u in range(A.shape[0]):
        # Loop through the elements above the diagonal
        for v in range(u + 1, A.shape[1]):

            # calculate 0-hop structual similarity
            g_0_neighbors_u = np.where(A[u] == 1)[0]
            g_0_neighbors_v = np.where(A[v] == 1)[0]
            degree_0_sequence_u = [degrees[i] for i in g_0_neighbors_u]
            degree_0_sequence_v = [degrees[i] for i in g_0_neighbors_v]
            degree_0_sequence_u.sort()
            degree_0_sequence_v.sort()
            if degree_0_sequence_u==[]:
                degree_0_sequence_u.append(0)
            if degree_0_sequence_v==[]:
                degree_0_sequence_v.append(0)
            # print(degree_0_sequence_u)
            # print(degree_0_sequence_v)
            w0, _, acc_cost_matrix, path = dtw(
                degree_0_sequence_u, degree_0_sequence_v, dist=d
            )
            s_0[u][v] = np.exp(-w0)

            # calculate 1-hop structual similarity
            g_1_neighbors_u = np.where(binary_A_2[u] == 1)[0]
            g_1_neighbors_v = np.where(binary_A_2[v] == 1)[0]
            degree_1_sequence_u = [degrees[i] for i in g_1_neighbors_u]
            degree_1_sequence_v = [degrees[i] for i in g_1_neighbors_v]
            degree_1_sequence_u.sort()
            degree_1_sequence_v.sort()
            if degree_1_sequence_u==[]:
                degree_1_sequence_u.append(0)
            if degree_1_sequence_v==[]:
                degree_1_sequence_v.append(0)
            w1, _, acc_cost_matrix, path = dtw(
                degree_1_sequence_u, degree_1_sequence_v, dist=d
            )
            s_1[u][v] = np.exp(-w1)

            # calculate 2-hop structual similarity
            g_2_neighbors_u = np.where(binary_A_3[u] == 1)[0]
            g_2_neighbors_v = np.where(binary_A_3[v] == 1)[0]
            degree_2_sequence_u = [degrees[i] for i in g_2_neighbors_u]
            degree_2_sequence_v = [degrees[i] for i in g_2_neighbors_v]
            degree_2_sequence_u.sort()
            degree_2_sequence_v.sort()
            if degree_2_sequence_u==[]:
                degree_2_sequence_u.append(0)
            if degree_2_sequence_v==[]:
                degree_2_sequence_v.append(0)
            w2, _, acc_cost_matrix, path = dtw(
                degree_2_sequence_u, degree_2_sequence_v, dist=d
            )
            s_2[u][v] = np.exp(-w2)

            # calculate 3-hop structual similarity
            g_3_neighbors_u = np.where(binary_A_4[u] == 1)[0]
            g_3_neighbors_v = np.where(binary_A_4[v] == 1)[0]
            degree_3_sequence_u = [degrees[i] for i in g_3_neighbors_u]
            degree_3_sequence_v = [degrees[i] for i in g_3_neighbors_v]
            degree_3_sequence_u.sort()
            degree_3_sequence_v.sort()
            if degree_3_sequence_u==[]:
                degree_3_sequence_u.append(0)
            if degree_3_sequence_v==[]:
                degree_3_sequence_v.append(0)
            w3, _, acc_cost_matrix, path = dtw(
                degree_3_sequence_u, degree_3_sequence_v, dist=d
            )
            s_3[u][v] = np.exp(-w3)
            # visualization check for dynamic time warping algorithm
            # print(np.exp(-w3))
            # plt.imshow(
            #     acc_cost_matrix.T, origin="lower", cmap="cividis", interpolation="nearest"
            # )
            # plt.plot(path[0], path[1], "w")
            # plt.show()
            # Copy the element from the upper triangular part to the lower triangular part
            s_0[v, u] = s_0[u, v]
            s_1[v, u] = s_1[u, v]
            s_2[v, u] = s_2[u, v]
            s_3[v, u] = s_3[u, v]
    if not need_identity:
        return s_0, s_1, s_2, s_3, A, binary_A_2, binary_A_3
    return s_0, s_1, s_2, s_3, A, binary_A_2, binary_A_3, I


if __name__ == "__main__":
    # input_root = '/mnt/disk1/HCP_luzhang_do/Analysis/Graph_embedding/Common_3hinge_results/Graph_embedding_data_200/adj_feature_matrix_200'
    input_root = "/media/minheng/hdd_3/HCP_cc_0819/HCP_new"
    
    subjects_list = [
        subject
        for subject in os.listdir(input_root)
        if not subject.startswith(".") and not subject.startswith("label")
    ]
    subjects_list = list(set(subjects_list))
    # print(subjects_list)
    # print(len(subjects_list))
    subjects_list.sort()
    sphere_list = ["lh", "rh"]
    # subjects_list = ["100408"]


    error_list=[]
    for subject in tqdm(subjects_list):
        save_path_1= input_root+ "/"+ str(subject)+ "/Graph_embedding_data"+ "/binary_structural_similarity"
        node_dir_1=  input_root+ "/"+ str(subject)+ "/Graph_embedding_data"+ "/bsse_node_input_data"
        create_or_clear_directory(save_path_1)
        create_or_clear_directory(node_dir_1)


        save_path_2= input_root+ "/"+ str(subject)+ "/Graph_embedding_data"+ "/structural_similarity"
        node_dir_2=  input_root+ "/"+ str(subject)+ "/Graph_embedding_data"+ "/sse_node_input_data"
        create_or_clear_directory(save_path_2)
        create_or_clear_directory(node_dir_2)
        
        for sphere in sphere_list:
            one_hot_feature = np.loadtxt(
                input_root
                + "/"
                + str(subject)
                + "/Graph_embedding_data"
                + "/adj_feature_matrix/"
                + str(subject)
                + "_3hinge_0_hop_feature_"
                + sphere
                + ".txt",
                dtype="int",
            )
            adj_1_hop = np.loadtxt(
                input_root
                + "/"
                + str(subject)
                + "/Graph_embedding_data"
                + "/adj_feature_matrix/"
                + str(subject)
                + "_3hinge_adj_"
                + sphere
                + "_1_hop.txt",
                dtype="int",
            )
            adj_2_hop = np.loadtxt(
                input_root
                + "/"
                + str(subject)
                + "/Graph_embedding_data"
                + "/adj_feature_matrix/"
                + str(subject)
                + "_3hinge_adj_"
                + sphere
                + "_2_hop.txt",
                dtype="int",
            )
            adj_3_hop = np.loadtxt(
                input_root
                + "/"
                + str(subject)
                + "/Graph_embedding_data"
                + "/adj_feature_matrix/"
                + str(subject)
                + "_3hinge_adj_"
                + sphere
                + "_3_hop.txt",
                dtype="int",
            )
            # detect isolated 3-hinge nodes
            # id_list = detect_error(adj_1_hop, sphere, subject)
            # id_list=id_list.tolist()
            # if id_list!=[]:
            #      for id in id_list:
            #         error_list.append([subject, sphere, id])
            
            s_0, s_1, s_2, s_3, A, A_2, A_3, I = calculate_structural_similarity(
                adj_1_hop
            )
            # save structural similarity using binary adjacent matrix
            sse_0_hop_feature = np.dot(s_0*I,one_hot_feature)
            sse_1_hop_feature = np.dot(s_1*A,one_hot_feature)
            sse_2_hop_feature = np.dot(s_2*A_2,one_hot_feature)
            sse_3_hop_feature = np.dot(s_3*A_3,one_hot_feature)
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_0_hop_feature_" + sphere + ".txt", sse_0_hop_feature, fmt="%.5f")
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_1_hop_feature_" + sphere + ".txt", sse_1_hop_feature, fmt="%.5f")
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_2_hop_feature_" + sphere + ".txt", sse_2_hop_feature, fmt="%.5f")
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_3_hop_feature_" + sphere + ".txt", sse_3_hop_feature, fmt="%.5f")
            for node in range(one_hot_feature.shape[0]):
                bsse_node_feature = np.zeros(shape=(4, one_hot_feature.shape[1]))
                bsse_node_feature[0, :] = sse_0_hop_feature[node, :]
                bsse_node_feature[1, :] = sse_1_hop_feature[node, :]
                bsse_node_feature[2, :] = sse_2_hop_feature[node, :]
                bsse_node_feature[3, :] = sse_3_hop_feature[node, :]
                np.savetxt(
                node_dir_1 + "/" + subject + '_' + sphere + "_" + str(node) + ".txt", bsse_node_feature, fmt="%.5f"
                )
            

            # save structural similarity using adjacent matrix
            sse_0_hop_feature=np.dot(s_0*I,one_hot_feature)
            sse_1_hop_feature=np.dot(s_1*adj_1_hop,one_hot_feature)
            sse_2_hop_feature=np.dot(s_2*adj_2_hop,one_hot_feature)
            sse_3_hop_feature=np.dot(s_3*adj_3_hop,one_hot_feature)
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_0_hop_feature_" + sphere + ".txt", sse_0_hop_feature, fmt="%.5f")
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_1_hop_feature_" + sphere + ".txt", sse_1_hop_feature, fmt="%.5f")
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_2_hop_feature_" + sphere + ".txt", sse_2_hop_feature, fmt="%.5f")
            np.savetxt(save_path_1 + "/" + str(subject) + "_3hinge_sse_3_hop_feature_" + sphere + ".txt", sse_3_hop_feature, fmt="%.5f")
            for node in range(one_hot_feature.shape[0]):
                node_feature = np.zeros(shape=(4, one_hot_feature.shape[1]))
                node_feature[0, :] = sse_0_hop_feature[node, :]
                node_feature[1, :] = sse_1_hop_feature[node, :]
                node_feature[2, :] = sse_2_hop_feature[node, :]
                node_feature[3, :] = sse_3_hop_feature[node, :]
                np.savetxt(
                node_dir_2 + "/" + subject + '_' + sphere + "_" + str(node) + ".txt", node_feature, fmt="%.5f"
                )
            # print(adj_1_hop)
            # pdb.set_trace()
    # error_list = pd.DataFrame(error_list)
    # # print(error_list)
    # header=['subject','sphere','id']
    # error_list.to_csv('error_3hinge.csv',header=header,index=False)
