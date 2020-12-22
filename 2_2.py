

import numpy as np
from Tree import Tree
from Tree import Node
from dynamicProg import DP

def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.
    """
    num_nodes = len(tree_topology)
    K = len(theta[0])

    u=3 #u can be any node in the tree, since we sum it out later
    DP_1 = DP(num_nodes,K,theta,tree_topology,beta)
    p_u = DP_1.joint_prob(u,beta)
    p_beta = sum(p_u) #summing joint prob p(X_u=i,beta) to obtain p(beta)

    print("Times called (S)",DP_1.timesCalledS," Times reused (S)", DP_1.counterS)
    print("Times called (T)",DP_1.timesCalledT," Times reused (T)", DP_1.counterT)
    print("Calculating the likelihood...")
    likelihood = p_beta
    return likelihood


def main():
    print("\n1. Load tree data from file and print it\n")

    #"data/q2_2/q2_2_small_tree.pkl", "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
    filename = "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))
    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx)#, "\tBeta: ", beta)
        print(len(t.get_topology_array()))
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
