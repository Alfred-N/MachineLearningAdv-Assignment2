import numpy as np

#Dynamic programming algorithm for calculating the joint probability p(X_u=i,beta|T,theta)
class DP:
    def __init__(self,num_nodes,K,theta,tree_topology,beta):
        self.t_ui = np.empty([num_nodes,K])
        self.t_ui[:] = float(np.nan)
        self.s_ui = np.empty([num_nodes,K])
        self.s_ui[:] = float(np.nan)
        self.K = K
        self.theta=theta
        self.beta=beta
        self.tree_topology=tree_topology
        self.childList = self.calc_Children()
        self.counterS=0
        self.counterT=0
        self.timesCalledS=0
        self.timesCalledT=0

    def calc_Children(self):
        num_nodes= len(self.tree_topology)
        childList = [[] for x in range(num_nodes)]
        for u in range(num_nodes-1,0,-1):
            curr_parent=int(self.tree_topology[u])
            childList[curr_parent].append(u)
        return childList

    #p(X_{o + ^u},X_u=i)
    def t_rec(self,u,i):
        self.timesCalledT+=1
        if np.isnan(self.t_ui[u,i]):
            if u == 0: #if root node
                t_ri = self.theta[0][i]
                self.t_ui[u][i]=t_ri
                return t_ri
            else:
                parent = int(self.tree_topology[u])
                siblings = self.childList[parent]
                for s in siblings:
                    if s!=u:
                        sib = int(s)
                t_sum = 0
                for j in range(self.K):#p(X_parent=j)
                    for k in range (self.K): #p(X_sibling = k)
                        t_sum += self.t_rec(parent,j)*self.theta[u][j][i]*self.theta[sib][j][k]*self.s_rec(sib,k)
                self.t_ui[u][i] = t_sum
                return t_sum
        else:
            self.counterT += 1
            return self.t_ui[u,i]
    
    #p(X_{o + <u},X_u=i)
    def s_rec(self,u,i):
        self.timesCalledS+=1
        if np.isnan(self.s_ui[u,i]):
            num_children = len(self.childList[u])
            if num_children == 0:
                if i == int(self.beta[u]):
                    s=1
                else:
                    s=0
                self.s_ui[u][i]= s
                return s
            else:
                facs = np.zeros(num_children)
                for c_idx,c in enumerate(self.childList[u]):
                    for j in range(self.K):
                        facs[c_idx]+=self.s_rec(int(c),j)*self.theta[int(c)][i][j]
                s = np.prod(facs)
                self.s_ui[u][i]=s
                return s

        else:
            self.counterS += 1
            return self.s_ui[u,i]

    def joint_prob(self,u,beta):
        p_u=np.zeros(self.K)
        for i in range(self.K):
            p_u[i] = self.t_rec(u,i) * self.s_rec(u,i)
        return p_u
