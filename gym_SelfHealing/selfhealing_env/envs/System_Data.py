"""Construct the system data for the SelfHealingEnv"""
import pandas as pd
import numpy as np

class System_Data():
    """System data struct for building environment """
    def __init__(
        self, 
        file_name:str, # Excel fire name
        Sb:float = 100, # MW/MVar/MVA
        V0:float = 1.0, # p.u.
        V_max:float = 1.05, # p.u.
        V_min:float = 0.95
    ) -> None: 
        """
        Args:
            file_name: the file name of the system data
            Sb: the base power of the system
            V0: p.u. voltage of the slack bus
            V_max\V_min: p.u. maximum\minimum nodal voltage
        """
        
        # Base values
        Vb = pd.read_excel(file_name,sheet_name='Base').to_numpy().item()
        # Ib = Sb/(math.sqrt(3)*Vb)   # kA
        Zb = Vb**2/Sb           # O
        
        # BigM
        Big_M_FF = 40 # BigM for commodity flow
        Big_M_V = 3 # BigM for voltage calculation
        BigM_SC = 2 # BigM for breaking multi SVC output solution
        
        # Read bus data
        Bus_Data=pd.read_excel(file_name,sheet_name='Bus').to_numpy()
        N_Bus = Bus_Data.shape[0]
        Pd_Data = Bus_Data[:, 1] / Sb
        Pd_Data_all = np.sum(Pd_Data)
        Qd_Data = Bus_Data[:, 2] / Sb
        Qd_Data_all = np.sum(Qd_Data)
        Pd_ratio = Pd_Data / Pd_Data_all  # Active load ratio
        Qd_ratio = Qd_Data / Qd_Data_all  # Reactive load ratio

        # Read branch data
        Branch_Data=pd.read_excel(file_name,sheet_name='Branch').to_numpy()
        N_Branch = Branch_Data.shape[0]
        N_TL = np.sum(Branch_Data[:, 6] == 1).item() # No. of tielines
        NT = N_TL # Here we let total time steps = No. of tielines
        N_NL = np.sum(Branch_Data[:, 6] == 0).item() # No. of non-tielines
        Branch_start = Branch_Data[:, 1].astype(int) # Starting/Ending node of branch
        Branch_end = Branch_Data[:, 2].astype(int)
        # LS_Mask = make_mask(N_Bus,N_Branch,Branch_start ) 
        # LE_Mask = make_mask(N_Bus,N_Branch,Branch_end )
        pIn = make_inc_matrix(Branch_start, Branch_end) # Node-branch incidence matrix  jk-ij
        # pInn = np.copy(pIn)
        # pInn[pInn > 0] = 0 # Negative part of I + [*]_ij
        R_Branch0 = Branch_Data[:, 3] / Zb
        X_Branch0 = Branch_Data[:, 4] / Zb
        # SZ_Branch0 = R_Branch0 ** 2 + X_Branch0 ** 2
        S_Branch0 = Branch_Data[:, 5] / Sb
        self.disturbance_set = np.where(Branch_Data[:, 7] == 1)[0].tolist() # Potential disturbance lines

        # Read DG data
        DG_Data = pd.read_excel(file_name,sheet_name='DG').to_numpy()
        N_DG = DG_Data.shape[0]
        DataDN_IndDG = DG_Data[:, 1]
        DataDN_IndBSDG = DG_Data[DG_Data[:, 6] == 1, 1] # List of black start DGs
        # DataDN_IndNMDG = DG_Data[DG_Data[:, 6] == 0, 1]
        N_BSDG = DataDN_IndBSDG.shape[0]
        # N_NMDG = DataDN_IndNMDG.shape[0]
        P_DG_max0 = DG_Data[:, 2] / Sb
        P_DG_min0 = DG_Data[:, 3] / Sb
        Q_DG_max0 = DG_Data[:, 4] / Sb
        Q_DG_min0 = DG_Data[:, 5] / Sb
        DG_Mask = make_mask(N_Bus, N_DG, DataDN_IndDG) # DG_Mask[i,j] = 1 if DG j is connected to bus i
        BSDG_Mask = make_mask(N_Bus, N_BSDG, DataDN_IndBSDG)
        # NMDG_Mask = make_mask(N_Bus, N_NMDG, DataDN_IndNMDG)

        # Set 
        load_pec = 1.0 # Load level
        Pd_all = Pd_Data_all * load_pec # Set loads matrix
        Qd_all = Qd_Data_all * load_pec
        Pd = Pd_all * np.tile(Pd_ratio, (NT, 1)).T
        Qd = Qd_all * np.tile(Qd_ratio, (NT, 1)).T
        R_Branch = np.tile(R_Branch0, (NT, 1)).T # Set branch matrix
        X_Branch = np.tile(X_Branch0, (NT, 1)).T
        S_Branch = np.tile(S_Branch0, (NT, 1)).T
        P_DG_max = np.tile(P_DG_max0, (NT, 1)).T # Set DG bounds matrix
        P_DG_min = np.tile(P_DG_min0, (NT, 1)).T
        Q_DG_max = np.tile(Q_DG_max0, (NT, 1)).T
        Q_DG_min = np.tile(Q_DG_min0, (NT, 1)).T
        
        # Key parameters for building opt models
        self.N_Bus = N_Bus
        self.N_Branch = N_Branch
        self.N_TL = N_TL
        self.NT = NT
        self.N_NL = N_NL
        self.S_lower_limit = -S_Branch0
        self.S_upper_limit = S_Branch0
        self.N_DG = N_DG
        self.Qsvc_lower_limit = Q_DG_min0[1:]
        self.Qsvc_upper_limit = Q_DG_max0[1:]
        self.Pd = Pd
        self.Qd = Qd
        self.Pd_all = Pd_all
        
        self.args_expert = (NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
                V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask,
                Big_M_FF)

        # NT=1 for step\reset model
        self.args_step = (1, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch[:, 0], X_Branch[:, 0], Big_M_V, V0,
                V_min, V_max, Pd[:, 0], Qd[:, 0], S_Branch[:, 0], P_DG_min[:, 0], P_DG_max[:, 0], Q_DG_min[:, 0],
                Q_DG_max[:, 0], BSDG_Mask, Big_M_FF)
        pass



# Utils for System_Data
def make_mask(
    x:int, 
    y:int, 
    a_list:np.ndarray
) -> np.ndarray:
    """Make a 2-D mask matrix for converting No. of y to No. of x according to the list a_list.

    Args:
        x (int): Number of x
        y (int): Number of y
        a_list (np.ndarray): a list which contants y rows and each row is a number from 1 to x

    Returns:
        mask (np.ndarray): the mask matrix for convert usage
                            mask(i,j)=1 means the jth element in a_list is i
    """
    mask = np.zeros((x, y))
    for i in range(x):
        mask[i, a_list == i+1] = 1
        
    return mask

def make_inc_matrix(
    start_node:np.ndarray, 
    end_node:np.ndarray
) -> np.ndarray:
    """Make the incidence matrix of node-branch

    Args:
        start_node (np.ndarray): starting nodes
        end_node (np.ndarray): ending nodes

    Returns:
        inc (np.ndarray): the incidence matrix 
                            inc[start_node[j],j]=1, inc[t[j],j]=-1 if line j is starting from start_node(j) to end_node(j)
    """
    max_node = max(np.max(start_node), np.max(end_node))
    inc = np.zeros((max_node, len(start_node)))
    for j in range(len(start_node)):
        inc[start_node[j]-1, j] = 1
        inc[end_node[j]-1, j] = -1
        
    return inc
