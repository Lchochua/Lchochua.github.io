import numpy as np
import matplotlib.pyplot as plt
import csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from pathlib import Path

# A - Functions

# A.1 - Function 'elementwise': An element-wise application of an operator to two matrices 
#
# Input: Operator operator, 'm times n'-Matrix M, 'm times n'-Matrix N
# Output: 'm times n'-Matrix A
def elementwise(operator, M, N):
    assert(M.shape == N.shape)
    nr, nc = M.shape[0], M.shape[1]
    A =  np.zeros((nr,nc))
    for r in range(nr):
        for c in range(nc):
            A[r,c] = operator(M[r,c], N[r,c])    
    return np.matrix(A)

# A.2 - Function 'Network': Construct the network matrix for each coalition 
#
# Input: Maximum Number nmax, Coalitions S
# Output: 'nmax times nmax'-Matrices A
def Network(nmax, S):
    # Step 1: Define useful arrays
    I = np.identity(nmax)
    z = np.zeros(nmax)
    # Step 2: Construct the network matrix for each coalition
    A = [np.matrix(np.identity(nmax)) for i in S]
    A[0] = np.matrix(np.ones((nmax,nmax)))
    A[1] = np.matrix(A[1]+[z,I[0],z,I[0],I[2],I[0],z,I[0],I[0]+I[5]+I[7],I[6],I[6],I[6]+I[9]+I[10],I[0],z,I[0],I[13]])
    A[2] = np.matrix(A[2]+[z,I[0],I[0],z,I[3],I[0],I[0],z,I[7],I[0]+I[5]+I[6],I[7],I[7]+I[8]+I[10],I[0],I[0],z,I[14]])
    A[3] = np.matrix(A[3]+[z,z,I[0],I[0],I[1],z,I[0],I[0],I[5],I[5],I[0]+I[6]+I[7],I[5]+I[8]+I[9],z,I[0],I[0],I[12]])
    A[4] = np.matrix(A[4]+[I[1]+I[5]+I[12],I[0]+I[5]+I[12],I[0]+I[1]+I[5]+I[12],I[0]+I[1]+I[5]+I[12],I[0]+I[1]+I[2]+I[3]+I[5]+I[12],I[0]+I[1]+I[12],I[0]+I[1]+I[5]+I[9]+I[12],I[0]+I[1]+I[5]+I[8]+I[12],I[0]+I[1]+I[5]+I[7]+I[12],I[0]+I[1]+I[5]+I[6]+I[12],I[0]+I[1]+I[5]+I[6]+I[7]+I[8]+I[9]+I[11]+I[12],I[0]+I[1]+I[5]+I[6]+I[7]+I[8]+I[9]+I[10]+I[12],I[0]+I[1]+I[5],I[0]+I[1]+I[5]+I[12],I[0]+I[1]+I[5]+I[12],I[0]+I[1]+I[5]+I[12]+I[13]+I[14]])
    A[5] = np.matrix(A[5]+[I[2]+I[6]+I[13],I[0]+I[2]+I[6]+I[13],I[0]+I[6]+I[13],I[0]+I[2]+I[6]+I[13],I[0]+I[1]+I[2]+I[3]+I[6]+I[13],I[0]+I[2]+I[6]+I[9]+I[13],I[0]+I[2]+I[13],I[0]+I[2]+I[6]+I[10]+I[13],I[0]+I[2]+I[5]+I[6]+I[7]+I[9]+I[10]+I[11]+I[13],I[0]+I[2]+I[5]+I[6]+I[13],I[0]+I[2]+I[6]+I[7]+I[13],I[0]+I[2]+I[5]+I[6]+I[7]+I[8]+I[9]+I[10]+I[13],I[0]+I[2]+I[6]+I[13],I[0]+I[2]+I[6],I[0]+I[2]+I[6]+I[13],I[0]+I[2]+I[6]+I[12]+I[13]+I[14]])
    A[6] = np.matrix(A[6]+[I[3]+I[7]+I[14],I[0]+I[3]+I[7]+I[14],I[0]+I[3]+I[7]+I[14],I[0]+I[7]+I[14],I[0]+I[1]+I[2]+I[3]+I[7]+I[14],I[0]+I[3]+I[7]+I[8]+I[14],I[0]+I[3]+I[7]+I[10]+I[14],I[0]+I[3]+I[14],I[0]+I[3]+I[5]+I[7]+I[14],I[0]+I[3]+I[5]+I[6]+I[7]+I[8]+I[10]+I[11]+I[14],I[0]+I[3]+I[6]+I[7]+I[14],I[0]+I[3]+I[5]+I[6]+I[7]+I[8]+I[9]+I[10]+I[14],I[0]+I[3]+I[7]+I[14],I[0]+I[3]+I[7]+I[14],I[0]+I[3]+I[7],I[0]+I[3]+I[7]+I[12]+I[13]+I[14]])
    return A

# A.3 - Function 'Pref': Compute the utility matrix for each player and consequentially determine the preference matrix for each coalition
#    
# Input: Players P, Coalitions S, Outcomes X, Endowments e, Factor alpha 
# Output: Utility Matrices U, Preference Matrices B
def Pref(P,S,X,e,alpha):
    # Step 1: Utilities 
    Util = [[0 for x in X] for p in P]
    for p in P:
        # Case 0: MFN   
        # Auxiliaries for MFN
        pme = (p)
        po1 = ((p+1) % 3)
        po2 = ((p+2) % 3)
        entry = 0
        # Utilities for MFN
        Util[pme][entry] = (-78*e[pme]**2+17*(e[po1]**2+e[po2]**2)+16*e[po1]*e[po2]+e[pme]*(256*alpha-30*(e[po1]+e[po2])))/128
        # Cases 1-3: CU
        # Subcases .1-.2: Insider of a CU       
        # Auxiliaries for the Insider of a CU 
        for v in range(2):
            pime=((p) % 3)
            pio=((p+1+((v) % 2)) % 3)
            poo=((p+1+((v+1) % 2)) % 3)
            entry=(1+((p+2*v) % 3))
            # Utilities for the Insider of CU(pime,pio)                             
            if e[poo] <= (13/11)*e[pime]:
                if e[poo] <= (13/11)*e[pio]: 
                    # Scenario: nobody capped 
                    Util[pime][entry] = (-1999*e[pime]**2+640*e[poo]**2+609*e[pio]**2-256*e[pio]*e[poo]+e[pime]*(6400*alpha-750*e[pio]-384*e[poo]))/3200
                if e[poo] > (13/11)*e[pio]:
                    # Scenario: pime capped [at MFN] 
                    Util[pime][entry] = (-17991*e[pime]**2+1750*e[pio]**2+3050*e[pio]*e[poo]+3901*e[poo]**2-18*e[pime]*(375*e[pio]+64*(-50*alpha+3*e[poo])))/28800
            if e[poo] > (13/11)*e[pime]:
                if e[poo] <= (13/11)*e[pio]: 
                    # Scenario: pio capped [at MFN]     
                    Util[pime][entry] = (-15950*e[pime]**2+5481*e[pio]**2+50*e[pime]*(1152*alpha-135*e[pio]-119*e[poo])-2304*e[pio]*e[poo]+6409*e[poo]**2)/28800
                if e[poo] > (13/11)*e[pio]:   
                    # Scenario: both capped [at MFN]
                    Util[pime][entry] = (-319*e[pime]**2+35*e[pio]**2+e[pime]*(1152*alpha-135*e[pio]-119*e[poo])+61*e[pio]*e[poo]+91*e[poo]**2)/576
        # Subcase .3: Outsider of a CU
        # Auxiliaries for the Outsider of a CU 
        pome=((p) % 3)
        pio1=((p+1) % 3)
        pio2=((p+2) % 3)
        entry=(1+((p+1) % 3))
        # Utilities for the Outsider of CU(pio1,pio2)
        if e[pome] <= (13/11)*e[pio1]:
            if e[pome] <= (13/11)*e[pio2]: 
                # Scenario: nobody capped
                Util[pome][entry] = (-336*e[pome]**2+33*(e[pio1]**2+e[pio2]**2)+32*e[pome]*(25*alpha-(e[pio1]+e[pio2]))+50*e[pio1]*e[pio2])/400
            if e[pome] > (13/11)*e[pio2]: 
                # Scenario: pio1 capped [at MFN]
                Util[pome][entry] = (594*e[pio1]**2+1075*e[pio2]**2+e[pome]*(-576*e[pio1]-1750*e[pio2]+14400*alpha-5399*e[pome])+900*e[pio1]*e[pio2])/7200
        if e[pome] > (13/11)*e[pio1]:
            if e[pome] <= (13/11)*e[pio2]: 
                # pio2 capped [at MFN]
                Util[pome][entry] = (594*e[pio2]**2+1075*e[pio1]**2+e[pome]*(-576*e[pio2]-1750*e[pio1]+14400*alpha-5399*e[pome])+900*e[pio2]*e[pio1])/7200
            if e[pome] > (13/11)*e[pio2]: 
                # both capped [at MFN]
                Util[pome][entry] = (43*(e[pio1]**2+e[pio2]**2)+e[pome]*(-70*(e[pio1]+e[pio2])+576*alpha)-190*e[pome]**2+36*e[pio1]*e[pio2])/288
        # Cases 5-7: FTA        
        # Subcases .1-.2: Insider of an FTA 
        # Auxiliaries for the Insider of an FTA  
        for v in range(2):
            pime=((p) % 3)
            pio=((p+1+((v) % 2)) % 3)
            poo=((p+1+((v+1) % 2)) % 3)
            entry=(5+((p+2*v) % 3))
            # Utilities for the Insider of FTA(pime,pio) 
            if e[poo] <= (43/29)*e[pime]:
                if e[poo] <= (43/29)*e[pio]:
                    if e[poo] >= (4/5)*e[pime]:
                        if e[poo] >= (4/5)*e[pio]:  
                            # Scenario: nobody capped                            
                            Util[pime][entry] = (-10159*e[pime]**2+3201*e[pio]**2+3072*e[poo]**2-1408*e[pio]*e[poo]+2*e[pime]*(15488*alpha-1815*e[pio]-768*e[poo]))/15488
                        if e[poo] < (4/5)*e[pio]:                             
                            # Scenario: pime capped at 0                            
                            Util[pime][entry] = -(10159/15488)*e[pime]**2+(145/1152)*e[pio]**2+e[pime]*(2*alpha-(15/64)*e[pio]-(12/121)*e[poo])+(e[pio]*e[poo])/9+(157/2178)*e[poo]**2                            
                    if e[poo] < (4/5)*e[pime]:   
                        if e[poo] >= (4/5)*e[pio]:                             
                            # Scenario: pio capped at 0                            
                            Util[pime][entry] = -(671/1152)*e[pime]**2+(291/1408)*e[pio]**2+e[pime]*(2*alpha-(15/64)*e[pio]-(2/9)*e[poo])-(e[pio]*e[poo])/11+(47/198)*e[poo]**2                            
                        if e[poo] < (4/5)*e[pio]:                             
                            # Scenario: both capped at 0                            
                            Util[pime][entry] = (-671*e[pime]**2+145*e[pio]**2+e[pime]*(-270*e[pio]+256*(9*alpha-e[poo]))+128*(e[pio]*e[poo]+e[poo]**2))/1152                            
                if e[poo] > (43/29)*e[pio]:                     
                    # Scenario: pime capped at MFN                    
                    Util[pime][entry] = (-91431*e[pime]**2+8470*e[pio]**2+18*e[pime]*(15488*alpha-1815*e[pio]-768*e[poo])+14762*e[pio]*e[poo]+18397*e[poo]**2)/139392                    
            if e[poo] > (43/29)*e[pime]:
                if e[poo] <= (43/29)*e[pio]:                     
                    # Scenario: pio capped at MFN                    
                    Util[pime][entry] = (-7018*e[pime]**2+2619*e[pio]**2+22*e[pime]*(1152*alpha-135*e[pio]-119*e[poo])-1152*e[pio]*e[poo]+2843*e[poo]**2)/12672                    
                if e[poo] > (43/29)*e[pio]:                     
                    # Scenario: both capped at MFN (cf. CU - Insider of a CU - both capped [at MFN])                    
                    Util[pime][entry] = (-319*e[pime]**2+35*e[pio]**2+e[pime]*(1152*alpha-135*e[pio]-119*e[poo])+61*e[pio]*e[poo]+91*e[poo]**2)/576
        # Subcase .3: Outsider of an FTA
        # Auxiliaries for the Outsider of an FTA          
        pome=((p) % 3)
        pio1=((p+1) % 3)
        pio2=((p+2) % 3)
        entry=(5+((p+1) % 3))     
        # Utilities for the Outsider of FTA(pio1,pio2)             
        if e[pome] <= (43/29)*e[pio1]:
            if e[pome] <= (43/29)*e[pio2]:
                if e[pome] >= (4/5)*e[pio1]:
                    if e[pome] >= (4/5)*e[pio2]:                         
                        # Scenario: nobody capped                        
                        Util[pome][entry] = (-1680*e[pome]**2+129*(e[pio1]**2+e[pio2]**2)+242*e[pio1]*e[pio2]+32*e[pome]*(121*alpha-2*(e[pio1]+e[pio2])))/1936                        
                    if e[pome] < (4/5)*e[pio2]:                        
                        # Scenario: pio1 capped at 0                        
                        Util[pome][entry] = (129/1936)*e[pio1]**2+(17/144)*e[pio2]**2+e[pio1]*(e[pio2]/8-(4/121)*e[pome])+2*alpha*e[pome]-(2/9)*e[pio2]*e[pome]-(775/1089)*e[pome]**2
                if e[pome] < (4/5)*e[pio1]:   
                    if e[pome] >= (4/5)*e[pio2]:                         
                        # Scenario: pio2 capped at 0                        
                        Util[pome][entry] = (129/1936)*e[pio2]**2+(17/144)*e[pio1]**2+e[pio2]*(e[pio1]/8-(4/121)*e[pome])+2*alpha*e[pome]-(2/9)*e[pio1]*e[pome]-(775/1089)*e[pome]**2
                    if e[pome] < (4/5)*e[pio2]:                         
                        # Scenario: both capped at 0                        
                        Util[pome][entry] = (17*(e[pio1]**2+e[pio2]**2)+2*e[pio1]*(9*e[pio2]-16*e[pome])-32*e[pio2]*e[pome]+16*(18*alpha-5*e[pome])*e[pome])/144                        
            if e[pome] > (43/29)*e[pio2]:                 
                # Scenario: pio1 capped at MFN                
                Util[pome][entry] = (2322*e[pio1]**2+5203*e[pio2]**2+36*e[pio1]*(121*e[pio2]-32*e[pome])-8470*e[pio2]*e[pome]+e[pome]*(69696*alpha-26615*e[pome]))/34848                
        if e[pome] > (43/29)*e[pio1]:
            if e[pome] <= (43/29)*e[pio2]:                 
                # Scenario: pio2 capped at MFN                
                Util[pome][entry] = (2322*e[pio2]**2+5203*e[pio1]**2+36*e[pio2]*(121*e[pio1]-32*e[pome])-8470*e[pio1]*e[pome]+e[pome]*(69696*alpha-26615*e[pome]))/34848                
            if e[pome] > (43/29)*e[pio2]:                 
                # Scenario: both capped at MFN (cf. CU - Outsider of a CU - both capped [at MFN])                
                Util[pome][entry] = (43*(e[pio1]**2+e[pio2]**2)+e[pome]*(-70*(e[pio1]+e[pio2])+576*alpha)-190*e[pome]**2+36*e[pio1]*e[pio2])/288
        # Cases 8-10: FTAHub          
        # Subcase .1: Hub of an FTAHub             
        # Auxiliaries for the Hub of an FTAHub                    
        pime=((p) % 3)
        poo1=((p+1) % 3)
        poo2=((p+2) % 3)
        entry=(8+((p) % 3)) 
        # Utilities for the Hub of FTAHub(pime)  
        if e[poo1] <= (43/29)*e[pime]:
            if e[poo2] <= (43/29)*e[pime]:
                if e[poo1] >= (4/5)*e[pime]:
                    if e[poo2] >= (4/5)*e[pime]:                         
                        # Scenario: nobody capped                        
                        Util[pime][entry] = (-1530*e[pime]**2+157*(e[poo1]**2+e[poo2]**2)+36*e[pime]*(121*alpha-6*(e[poo1]+e[poo2]))+242*e[poo1]*e[poo2])/2178                        
                    if e[poo2] < (4/5)*e[pime]:                         
                        # Scenario: poo1 capped at 0                        
                        Util[pime][entry] = (-1370*e[pime]**2+157*e[poo1]**2+242*e[poo1]*e[poo2]+242*e[poo2]**2-4*e[pime]*(54*e[poo1]+121*(-9*alpha+e[poo2])))/2178                        
                if e[poo1] < (4/5)*e[pime]:   
                    if e[poo2] >= (4/5)*e[pime]:                         
                        # Scenario: poo2 capped at 0                        
                        Util[pime][entry] = (-1370*e[pime]**2+157*e[poo2]**2+242*e[poo2]*e[poo1]+242*e[poo1]**2-4*e[pime]*(54*e[poo2]+121*(-9*alpha+e[poo1])))/2178                        
                    if e[poo2] < (4/5)*e[pime]:                         
                        # Scenario: both capped at 0 (cf. GFT)                        
                        Util[pime][entry] = (-5*e[pime]**2+e[poo1]**2+e[poo2]**2+e[poo1]*e[poo2]+2*e[pime]*(9*alpha-e[poo1]-e[poo2]))/9                        
            if e[poo2] > (43/29)*e[pime]:                 
                # Scenario: poo1 capped at MFN                
                Util[pime][entry] = (-83687*e[pime]**2+10048*e[poo1]**2+2*e[pime]*(139392*alpha-6912*e[poo1]-14399*e[poo2])+15488*e[poo1]*e[poo2]+13673*e[poo2]**2)/139392                
        if e[poo1] > (43/29)*e[pime]:
            if e[poo2] <= (43/29)*e[pime]:                 
                # Scenario: poo2 capped at MFN                
                Util[pime][entry] = (-83687*e[pime]**2+10048*e[poo2]**2+2*e[pime]*(139392*alpha-6912*e[poo2]-14399*e[poo1])+15488*e[poo2]*e[poo1]+13673*e[poo1]**2)/139392                
            if e[poo2] > (43/29)*e[pime]:                 
                # Scenario: both capped at MFN                
                Util[pime][entry] = (-574*e[pime]**2+113*(e[poo1]**2+e[poo2]**2)+e[pime]*(2304*alpha-238*(e[poo1]+e[poo2]))+128*e[poo1]*e[poo2])/1152       
        # Subcases .2-3: Part (not Hub) of an FTAHub
        # Auxiliaries for the Parts (not Hub) of an FTAHub         
        for v in range(2):
            pome=((p) % 3)
            pio=((p+1+((v) % 2)) % 3)
            poo=((p+1+((v+1) % 2)) % 3)
            entry=(8+((p+1+v) % 3)) 
            # Utilities for the Parts (not Hub) of FTAHub(pio)
            if e[pome] <= (43/29)*e[pio]:
                if e[poo] <= (43/29)*e[pio]:
                    if e[pome] >= (4/5)*e[pio]:
                        if e[poo] >= (4/5)*e[pio]:                             
                            # Scenario: nobody capped                            
                            Util[pome][entry] = (-1550*e[pome]**2+306*e[pio]**2+517*e[poo]**2-198*e[pio]*e[poo]+e[pome]*(-72*e[pio]+484*(9*alpha-e[poo])))/2178                            
                        if e[poo] < (4/5)*e[pio]:                             
                            # Scenario: pome capped at 0                            
                            Util[pome][entry] = (65*e[pio]**2-775*e[pome]**2+242*e[pome]*(9*alpha-e[poo])+121*e[poo]**2+e[pio]*(-36*e[pome]+121*e[poo]))/1089                            
                    if e[pome] < (4/5)*e[pio]:   
                        if e[poo] >= (4/5)*e[pio]:                             
                            # Scenario: poo capped at 0                            
                            Util[pome][entry] = (38*e[pio]**2+47*e[poo]**2-44*e[poo]*e[pome]+22*(18*alpha-5*e[pome])*e[pome]-2*e[pio]*(9*e[poo]+22*e[pome]))/198                            
                        if e[poo] < (4/5)*e[pio]:                             
                            # Scenario: both capped at 0 (cf. GFT)                            
                            Util[pome][entry] = (-5*e[pome]**2+e[pio]**2+e[poo]**2+e[pio]*e[poo]+2*e[pome]*(9*alpha-e[pio]-e[poo]))/9                            
                if e[poo] > (43/29)*e[pio]:                     
                    # Scenario: pome capped at MFN                    
                    Util[pome][entry] = -(755/139392)*e[pio]**2-(775/1089)*e[pome]**2+e[pome]*(2*alpha-(2/9)*e[poo])+e[pio]*(-(4/121)*e[pome]+(61/576)*e[poo])+(197/1152)*e[poo]**2                    
            if e[pome] > (43/29)*e[pio]:
                if e[poo] <= (43/29)*e[pio]:                     
                    # Scenario: poo capped at MFN                    
                    Util[pome][entry] = (707*e[pio]**2+752*e[poo]**2-704*e[poo]*e[pome]+11*(576*alpha-175*e[pome])*e[pome]-2*e[pio]*(144*e[poo]+385*e[pome]))/3168                    
                if e[poo] > (43/29)*e[pio]:                     
                    # Scenario: both capped at MFN                    
                    Util[pome][entry] = (89*e[pio]**2-700*e[pome]**2+256*e[pome]*(9*alpha-e[poo])+197*e[poo]**2+e[pio]*(-280*e[pome]+122*e[poo]))/1152
        # Cases 12-14: MTA
        # Subcases .1-.2: Insider of an MTA 
        # Auxiliaries for the Insider of an MTA 
        for v in range(2):
            pime=((p) % 3)
            pio=((p+1+((v) % 2)) % 3)
            poo=((p+1+((v+1) % 2)) % 3)
            entry=(12+((p+2*v) % 3))             
            # Utilities for the Insider of MTA(pime,pio)            
            Util[pime][entry] = (-3447*e[pime]**2+633*e[pio]**2+896*e[poo]**2+1024*e[pio]*e[poo]+2*e[pime]*(6272*alpha-735*e[pio]-960*e[poo]))/6272
        # Subcase .3: Outsider of an MTA 
        # Auxiliaries for the Outsider of an MTA 
        pome=((p) % 3)
        pio1=((p+1) % 3)
        pio2=((p+2) % 3)
        entry=(12+((p+1) % 3))
        # Utilities for the Outsider of MTA(pio1,pio2)
        Util[pome][entry] = (-528*e[pome]**2+81*(e[pio1]**2+e[pio2]**2)+98*e[pio1]*e[pio2]+32*e[pome]*(49*alpha-4*(e[pio1]+e[pio2])))/784
        # Cases 4,11,15: GFT
        # Auxiliaries for GFT        
        pme = (p)
        po1 = ((p+1) % 3)
        po2 = ((p+2) % 3)
        entryCU = 4
        entryFTA = 11
        entryMult = 15
        welfareGFT = (-5*e[pme]**2+e[po1]**2+e[po2]**2+e[po1]*e[po2]+2*e[pme]*(9*alpha-e[po1]-e[po2]))/9 
        # Utilities for GFTCU [Case 4]        
        Util[pme][entryCU] = welfareGFT        
        # Utilities for GFTFTA [Case 11]        
        Util[pme][entryFTA] = welfareGFT         
        # Utilities for GFTMTA [Case 15]              
        Util[pme][entryMult] = welfareGFT
    # Step 2: Preferences
    B = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
    for p in P:
        for i in X:
            for j in X:
                B[p+1][i,j]= np.greater(Util[p][j],Util[p][i])            
    B[4] = elementwise(np.multiply, B[1], B[2])
    B[5] = elementwise(np.multiply, B[2], B[3])
    B[6] = elementwise(np.multiply, B[3], B[1])
    B[0] = elementwise(np.multiply, B[4], B[3])
    # Step 3: Utilities with Preferences
    R = [Util,B]
    return R

# A.4 - Function 'Search': Finds the Stable Set (where the stability depends on a parameter)
#    
# Input: Coalitions S, Outcomes X, Network A, Preferences B, Parameter o 
# Output: Stable Set x
#
# Note: The parameter 'o' determines whether the search algorithm considers
# deviations from deviations with infinite (intermediate) steps (o = 0) or
# with o (intermediate) steps (o >= 1) or computes the undominated outcomes
# under direct dominance (o = -1)
def Search(S,X,A,B,o):
    # Step 1: 'Direct Dominance'
    C = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
    for s in S:
        C[s] = elementwise(np.multiply, A[s], B[s])
    D = np.matrix(np.zeros((len(X),len(X))))
    for s in S:
        D = elementwise(np.logical_or, D, C[s])
    # Step 2: 'Indirect Dominance(s)'
    if o >= 0:
        if o >= 1:
            ocount = o
        if o == 0:
            ocount = 1
        while ocount >= 1:
            AD = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
            E = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
            for s in S:
                AD[s] = np.matrix([[1-np.prod([(1-A[s][i,z]*D[z,j]) for z in X]) for j in X] for i in X])
                E[s] = elementwise(np.multiply, elementwise(np.logical_or, A[s], AD[s]), B[s])
            F = np.matrix(np.zeros((len(X),len(X))))
            for s in S:
                F = elementwise(np.logical_or, F, E[s])
            if (F==D).all():
                break
            else: 
                D = F
                if o >= 1:
                    ocount = ocount - 1
    # Step 3: Find the Stable Set
    x = np.ones(len(X))
    xnext = np.zeros(len(X))
    if o >= 0:
        while True:
            for i in X:
                if x[i] == 0:
                    xnext[i] = 0
                else:
                    xneg = 0
                    for l in S:
                        for k in X:
                            xneghelp = A[l][i,k]*np.prod([(1-x[z]*(np.logical_or(np.equal(k,z),D[k,z]))*(1-B[l][i,z])) for z in X])
                            xneg = np.logical_or(xneg,xneghelp)
                    xnext[i] = x[i]-xneg
            if (xnext==x).all():
                break
            else:
                x = xnext
        return [x,D]
    if o == -1:
        for i in X:
            xneg = 0
            for j in X:
                xneg = np.logical_or(xneg,D[i,j])
            x[i] = x[i]-xneg
        return [x,D]

# A.5 - Function 'Endowments': Matches the Set of Endowments with the Stable Outcomes (where the stability depends on a parameter)
#    
# Input: Set of Endowments E, Maximal Number of Outcomes nmax, Players P, Coalitions S, Factor alpha, Parameter o, Trade Agreement Configuration var
# Output: Categorized Endowments for the Stable Outcomes E2X, Collection of Endowments, Stable Sets, and Utilities ExU, Collection of Endowments and Indirect Dominance Matrices ExD
#
# Note: The parameter 'o' determines whether the search algorithm considers
# deviations from deviations with infinite (intermediate) steps (o = 0) or
# with o (intermediate) steps (o >= 1) or computes the undominated outcomes
# under direct dominance (o = -1)
def Endowments2StableOutcomes(E,nmax,P,S,alpha,o,var):
    # Step 1a: Define the modified set of outcomes    
    n = 1+var[0]*4+var[1]*7+var[2]*4
    X = range(n)
    # Step 1b: Compile the modified list of outcomes for exclusion    
    LCU = []
    LFTA = []
    LMult = []
    if var[0] == 0:
        LCU = [1,2,3,4]
    if var[1] == 0:
        LFTA = [5,6,7,8,9,10,11]
    if var[2] == 0:
        LMult = [12, 13, 14, 15]
    L = LCU + LFTA + LMult
    # Step 1c: Construct the modified network matrices
    A = Network(nmax, S)
    if var[0]*var[1]*var[2] == 0:
        for s in S:
            A[s] = np.delete(A[s],L,0)    
            A[s] = np.delete(A[s],L,1)   
    # Step 2: Compute the stable sets for the representative set of endowments
    E2Xx = [[] for i in range(11)]
    E2Xy = [[] for i in range(11)]    
    ExU = []
    ExD = []
    for e in E:
        R = Pref(P,S,range(nmax),e,alpha)
        U = R[0]
        B = R[1]
        if var[0]*var[1]*var[2] == 0:
            for s in S:
                B[s] = np.delete(B[s],L,0)
                B[s] = np.delete(B[s],L,1)                     
        y0 = Search(S,X,A,B,0)[0]
        if y0[0] == 1:
            E2Xx[0] += [e[0]]
            E2Xy[0] += [e[2]]
        if var[0] == 1:
            if (1-y0[1])*(1-y0[2])*(1-y0[3]) == 0:
                E2Xx[1] += [e[0]]
                E2Xy[1] += [e[2]]
            if y0[4] == 1:
                E2Xx[2] += [e[0]]
                E2Xy[2] += [e[2]]
        if var[1] == 1:
            if (1-y0[5-(1-var[0])*4])*(1-y0[6-(1-var[0])*4])*(1-y0[7-(1-var[0])*4]) == 0:
                E2Xx[3] += [e[0]]
                E2Xy[3] += [e[2]]                               
            if (1-y0[8-(1-var[0])*4])*(1-y0[9-(1-var[0])*4])*(1-y0[10-(1-var[0])*4]) == 0:
                E2Xx[4] += [e[0]]
                E2Xy[4] += [e[2]]
                if y0[8-(1-var[0])*4] == 1:
                    if e[0] > (5/4)*max(e[1],e[2]):
                        E2Xx[8] += [e[0]]
                        E2Xy[8] += [e[2]]                     
                if y0[9-(1-var[0])*4] == 1:
                    if e[1] > (5/4)*max(e[0],e[2]):
                        E2Xx[8] += [e[0]]
                        E2Xy[8] += [e[2]]   
                if y0[10-(1-var[0])*4] == 1:
                    if e[2] > (5/4)*max(e[0],e[1]):
                        E2Xx[8] += [e[0]]
                        E2Xy[8] += [e[2]]                   
            if y0[11-(1-var[0])*4] == 1:
                E2Xx[5] += [e[0]]
                E2Xy[5] += [e[2]]        
        if var[2] == 1:
            if (1-y0[12-(1-var[0])*4-(1-var[1])*7])*(1-y0[13-(1-var[0])*4-(1-var[1])*7])*(1-y0[14-(1-var[0])*4-(1-var[1])*7]) == 0:
                E2Xx[6] += [e[0]]
                E2Xy[6] += [e[2]] 
            if y0[15-(1-var[0])*4-(1-var[1])*7] == 1:
                E2Xx[7] += [e[0]]
                E2Xy[7] += [e[2]]       
        if var[0]*var[1] == 1:
            if (1-y0[1])*(1-y0[2])*(1-y0[3])*(1-y0[5])*(1-y0[6])*(1-y0[7])*(1-y0[8])*(1-y0[9])*(1-y0[10]) == 0:
                E2Xx[9] += [e[0]]
                E2Xy[9] += [e[2]]        
            if (1-y0[4])*(1-y0[11])*(1-y0[15]) == 0:
                E2Xx[10] += [e[0]]
                E2Xy[10] += [e[2]]       
        ExU += [[float(e[0]),float(e[1]),float(e[2]),float(alpha)]+[y0[i] for i in range(0,len(y0))]]
        for p in P:   
            ExU += [[float(e[0]),float(e[1]),float(e[2]),float(alpha)]+[float(U[p][i]) for i in range(nmax)]]
        D = Search(S,X,A,B,0)[1]
        ExD += [[float(e[0]),float(e[1]),float(e[2])]]
        for l in range(n):
            ExD += [[int(D[l,i]) for i in range(n)]]
    E2X = [E2Xx,E2Xy]
    return [E2X,ExU,ExD]

# A.6 - Function 'AnalysisGraphicsFileGeneration': Provides the specific graphics file destination
#    
# Input: Path to the Analysis Folder AnalysisPath, Trade Agreement Configuration var, Case i
# Output: Graphics File Destination graphicsFile
def AnalysisGraphicsFileGeneration(AnalysisPath,var,i):
    if var == [1,1,1]:
        TempPath = AnalysisPath / 'General'
        if i == 0:
            graphicsFile = TempPath / 'general_MFN.png'
        elif i == 1:
            graphicsFile = TempPath / 'general_CU.png'
        elif i == 2:
            graphicsFile = TempPath / 'general_GFTCU.png'
        elif i == 3:
            graphicsFile = TempPath / 'general_FTA.png'
        elif i == 4:
            graphicsFile = TempPath / 'general_FTAHub.png'
        elif i == 5:
            graphicsFile = TempPath / 'general_GFTFTA.png'
        elif i == 6:
            graphicsFile = TempPath / 'general_M.png'
        elif i == 7:
            graphicsFile = TempPath / 'general_GFTM.png'
        elif i == 9:
            graphicsFile = TempPath / 'general_PTA.png'
        elif i == 10:
            graphicsFile = TempPath / 'general_GFT.png'
    elif var == [0,1,1]:
        TempPath = AnalysisPath / 'noCU'
        if i == 0:
            graphicsFile = TempPath / 'noCU_MFN.png'
        elif i == 3:
            graphicsFile = TempPath / 'noCU_FTA.png'
        elif i == 4:
            graphicsFile = TempPath / 'noCU_FTAHub.png'
        elif i == 5:
            graphicsFile = TempPath / 'noCU_GFTFTA.png'
        elif i == 6:
            graphicsFile = TempPath / 'noCU_M.png'
        elif i == 7:
            graphicsFile = TempPath / 'noCU_GFTM.png'
    elif var == [1,0,1]:            
        TempPath = AnalysisPath / 'noFTA'
        if i == 0:
            graphicsFile = TempPath / 'noFTA_MFN.png'
        elif i == 1:
            graphicsFile = TempPath / 'noFTA_CU.png'
        elif i == 2:
            graphicsFile = TempPath / 'noFTA_GFTCU.png'
        elif i == 6:
            graphicsFile = TempPath / 'noFTA_M.png'
        elif i == 7:
            graphicsFile = TempPath / 'noFTA_GFTM.png' 
    elif var == [0,0,1]:
        TempPath = AnalysisPath / 'noFTAaCU'
        if i == 0:
            graphicsFile = TempPath / 'noFTAaCU_MFN.png'
        elif i == 6:
            graphicsFile = TempPath / 'noFTAaCU_M.png'
        elif i == 7:
            graphicsFile = TempPath / 'noFTAaCU_GFTM.png' 
    else: 
        TempPath = AnalysisPath 
        if i == 0:
            graphicsFile = TempPath / 'graphics_MFN.png'
        elif i == 1:
            graphicsFile = TempPath / 'graphics_CU.png'
        elif i == 2:
            graphicsFile = TempPath / 'graphics_GFTCU.png'
        elif i == 3:
            graphicsFile = TempPath / 'graphics_FTA.png'
        elif i == 4:
            graphicsFile = TempPath / 'graphics_FTAHub.png'
        elif i == 5:
            graphicsFile = TempPath / 'graphics_GFTFTA.png'
        elif i == 6:
            graphicsFile = TempPath / 'graphics_M.png'
        elif i == 7:
            graphicsFile = TempPath / 'graphics_GFTM.png'
    return graphicsFile

# A.7 - Function 'Endowments': Determines the relevant endowments
#    
# Input: Minimal Endowment Factor facmin, Maximal Endowment Factor facmax, Endowment Type EType, Number of Steps H
# Output: Endowments E
def Endowments(facmin,facmax,EType,H):
    E = []
    g = (facmax-facmin)/(H-1)
    if EType == 'QR':
        for h in range(H):
            E += [[facmax, 1, facmin+h*g]]
    elif EType == 'PQ':
        for h in range(H):
            E += [[facmin+h*g, 1, 1]]
    elif EType == 'PR':
        for h in range(H):
            E += [[facmin+h*g, 1, facmin+h*g]]
    elif EType == 'A':
        for h1 in range(H):
            for h2 in range(h1+1):
                E += [[facmin+h1*g, 1, facmin+h2*g]]
    return E

# A.8 - Function 'UniversalAlpha': Determines the (unviersal) reservation price
#    
# Input: Endowments E, Players P, Precision epsilon
# Output: Reservation Price alpha
def UniversalAlpha(E,P,epsilon):
    alphamin=0
    for e in E:
        T = [0 for q in range(21)]
        i = 3
        for p in P:
            T[p]=5*(e[p]+e[(p+1) % 3])/12
            for q in P:
                if (p == q) == False:
                    T[i]=(3*e[p]+1*e[q])/5
                    T[i+6]=(1*e[p]+7*e[q])/11
                    T[i+12]=(2*e[p]+3*e[q])/7
                    i = i+1
        alphamintemp = max(T)
        alphamin = max([alphamintemp,alphamin])
    return alphamin+epsilon

# A.9 - Function 'Endowments_Alternative': Determines the relevant alternative endowments
#    
# Input: Point a point_a, Point b point_b, Point c point_c, Number of Steps H
# Output: Alternative Endowments E
def Alternative_Endowments(point_a,point_b,point_c,H):
    E = []
    ETemp = []
    xmin = min(point_a[0],point_b[0],point_c[0])
    xmax = max(point_a[0],point_b[0],point_c[0])
    ymin = min(point_a[1],point_b[1],point_c[1])
    ymax = max(point_a[1],point_b[1],point_c[1])
    xstep = (xmax-xmin)/(H-1)
    ystep = (ymax-ymin)/(H-1)
    for h1 in range(H):
        for h2 in range(h1+1):
            ETemp += [[xmin+h1*xstep, 1, ymin+h2*ystep]]
    Triangle = Polygon([point_a,point_b,point_c])
    for e in ETemp:
        ThisPoint = Point((e[0],e[2]))
        if Triangle.contains(ThisPoint) == True:
            E += [e]
    return E

# B - Program

# Part 0: Establishing the type of analysis
defaultCheck = input('Do you want to run the default? ("yes" or "no"): ')
if defaultCheck == 'yes':
    EType = 'A'
    V = [[1,1,1],[0,1,1],[1,0,1],[0,0,1]]
elif defaultCheck == 'no':  
    while True:
        EType = input('Choose between the edges "PQ", "QR", "PR", the full area "A", the alternative triangle "T": ') 
        if EType == 'PQ' or EType == 'QR' or EType == 'PR' or EType == 'A':
            break
        if EType == 'T':
            point_a_x = float(input('Point a (x): ')) 
            point_a_y = float(input('Point a (y): ')) 
            point_a = [point_a_x,point_a_y]
            point_b_x = float(input('Point b (x): ')) 
            point_b_y = float(input('Point b (y): ')) 
            point_b = [point_b_x,point_b_y]
            point_c_x = float(input('Point c (x): ')) 
            point_c_y = float(input('Point c (y): ')) 
            point_c = [point_c_x,point_c_y]
            break
    V = []    
    while True:
        vType1 = input('Do you want to include CU? ("yes" or "no"): ')
        vType2 = input('Do you want to include FTA? ("yes" or "no"): ')
        vType3 = input('Do you want to include MTA? ("yes" or "no"): ')
        vType = [vType1,vType2,vType3]
        vVec = [0,0,0]
        for i in range(3):
            if vType[i] == 'yes':
                vVec[i] = 1
        V += [vVec]
        vCheck = input('Do you want to add another case? ("yes" or "no"): ')
        if vCheck == 'no':
            break

# Part 1: Settings
# Set the number of steps for the endowments 'H' 
H = 500 # H=500
# Set the calculation precision 'epsilon'
epsilon=1/100
# Set the maximal / minimal multiplicative factor 'facmax' / 'facmin' 
facmin = 1
facmax = 5/3
# Define the set of players 'P' and coalitions 'S'
P = range(3) # 0 = player 1, 1 = player 2, 2 = player 3
S = range(7) # 0 = {1,2,3}, 1 = {1}, 2 = {2}, 3 = {3}, 4 = {1,2}, 5 = {2,3}, 6 = {3,1}
# Fix the number of maximal outcomes 'nmax'
nmax = 16 # 0 = empty, 1 = CU(12), 2 = CU(23), 3 = CU(31), 4 = CU(123), 5 = FTA(12), 6 = FTA(23), 7 = FTA(31), 8 = FTA(12,31), 9 = FTA(12,23), 10 = FTA(23,31), 11 = FTA(123), 12 = M(1,2), 13 = M(2,3), 14 = M(3,1), 15 = M(1,2,3)

# Part 2: Computation
# Determine the relevant endowments and the minimal (universal) reservation price
if EType == 'T':
    E = Alternative_Endowments(point_a,point_b,point_c,H)
else:
    E = Endowments(facmin,facmax,EType,H)
alpha = UniversalAlpha(E,P,epsilon)
# Prepare the plot(s)
figurenumber = 1
poly = Polygon([(1,1),(5/3,5/3),(5/3,1),(1,1)])
polyx,polyy = poly.exterior.xy
plt.figure(figurenumber)
plt.close()	
# Consider different combinations (or modifications) of Trade Agreements
for var in V:    
    # Matches the Set of Endowments with the Stable Outcomes
    metaSet = Endowments2StableOutcomes(E,nmax,P,S,alpha,0,var)
    # Now, plot the graphics and populate the data files with the Stable Sets
    AnalysisPath = Path(__file__).resolve().parent / 'Analysis' 
    if var == [1,1,1]:
        AnalysisFile = AnalysisPath / 'General' / 'results_general.csv'
        RelationsFile = AnalysisPath / 'General' / 'relations_general.csv'
        for i in range(11):
            if i != 8:
                plt.figure(figurenumber)
                plt.scatter(metaSet[0][0][i],metaSet[0][1][i], color = "gray", s=1)
                if i == 4:
                    plt.scatter(metaSet[0][0][8],metaSet[0][1][8], color = "red", s=1)
                plt.plot(polyx, polyy, color='black', alpha=1, linewidth=1, solid_capstyle='round', zorder=2)
                plt.xlabel('$e_1$')
                plt.ylabel('$e_2$')
                plt.savefig(AnalysisGraphicsFileGeneration(AnalysisPath,var,i))
                plt.close()	
                figurenumber = figurenumber + 1
    elif var == [0,1,1]:
        AnalysisFile = AnalysisPath / 'noCU' / 'results_noCU.csv'
        RelationsFile = AnalysisPath / 'noCU' / 'relations_noCU.csv'
        for i in range(8):
            if i != 1 and i != 2:
                plt.figure(figurenumber)
                plt.scatter(metaSet[0][0][i],metaSet[0][1][i], color = "gray", s=1)
                if i == 4:
                    plt.scatter(metaSet[0][0][8],metaSet[0][1][8], color = "red", s=1)
                plt.plot(polyx, polyy, color='black', alpha=1, linewidth=1, solid_capstyle='round', zorder=2)
                plt.xlabel('$e_1$')
                plt.ylabel('$e_2$')                
                plt.savefig(AnalysisGraphicsFileGeneration(AnalysisPath,var,i))
                plt.close()	
                figurenumber = figurenumber + 1
    elif var == [1,0,1]:
        AnalysisFile = AnalysisPath / 'noFTA' / 'results_noFTA.csv'
        RelationsFile = AnalysisPath / 'noFTA' / 'relations_noFTA.csv'
        for i in range(8):
            if i != 3 and i != 4 and i != 5:
                plt.figure(figurenumber)
                plt.scatter(metaSet[0][0][i],metaSet[0][1][i], color = "gray", s=1)
                plt.plot(polyx, polyy, color='black', alpha=1, linewidth=1, solid_capstyle='round', zorder=2)
                plt.xlabel('$e_1$')
                plt.ylabel('$e_2$')
                plt.savefig(AnalysisGraphicsFileGeneration(AnalysisPath,var,i))
                plt.close()	
                figurenumber = figurenumber + 1 
    elif var == [0,0,1]:
        AnalysisFile = AnalysisPath / 'noFTAaCU' / 'results_noFTAaCU.csv'
        RelationsFile = AnalysisPath / 'noFTAaCU' / 'relations_noFTAaCU.csv'
        for i in range(8):
            if i != 1 and i != 2 and i != 3 and i != 4 and i != 5:
                plt.figure(figurenumber)
                plt.scatter(metaSet[0][0][i],metaSet[0][1][i], color = "gray", s=1)
                plt.plot(polyx, polyy, color='black', alpha=1, linewidth=1, solid_capstyle='round', zorder=2)
                plt.xlabel('$e_1$')
                plt.ylabel('$e_2$')
                plt.savefig(AnalysisGraphicsFileGeneration(AnalysisPath,var,i))
                plt.close()	
                figurenumber = figurenumber + 1 
    else:
        AnalysisFile = AnalysisPath / 'results.csv'           
        RelationsFile = AnalysisPath / 'relations.csv'
        I = [0]
        if var[0] == 1:
            I += [1,2] 
        if var[1] == 1:
            I += [3,4,5]
        if var[2] == 1:
            I += [6,7]   
        for i in I:
            plt.figure(figurenumber)
            plt.scatter(metaSet[0][0][i],metaSet[0][1][i], color = "gray", s=1)
            if i == 4:
                plt.scatter(metaSet[0][0][8],metaSet[0][1][8], color = "red", s=1)
            plt.plot(polyx, polyy, color='black', alpha=1, linewidth=1, solid_capstyle='round', zorder=2)
            plt.xlabel('$e_1$')
            plt.ylabel('$e_2$')
            plt.savefig(AnalysisGraphicsFileGeneration(AnalysisPath,var,i))
            plt.close()	
            figurenumber = figurenumber + 1                    
    with open(AnalysisFile, mode='w+') as csvfile:
        csvfilewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for TempElement in metaSet[1]:
            csvfilewriter.writerow(TempElement)  
    with open(RelationsFile, mode='w+') as csvfile:
        csvfilewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for TempElement in metaSet[2]:
            csvfilewriter.writerow(TempElement)                
    # Announce the case and its computational progress
    print(var)
    print('Done!')