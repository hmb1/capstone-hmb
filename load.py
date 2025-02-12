import numpy as np

def load_function(idx, add=0):
    X = np.load(f'function_{idx}/initial_inputs.npy')
    y = np.load(f'function_{idx}/initial_outputs.npy')

    if add:
        X_n, y_n = add_points(idx,add )       
        
        X =np.vstack((X,X_n))
        # remember y is one dimensional
        y = np.append(y, y_n, axis=0)

    return (X,y)

def add_points(idx,add=1):
        y = []
        X = np.array([])


        if add >=1: 
            X1 =  [np.array([0.651, 0.68 ]), np.array([0.7  , 0.926]), np.array([0.493, 0.612, 0.34 ]), np.array([0.578, 0.42 , 0.426, 0.249]), np.array([0.22 , 0.846, 0.879, 0.88 ]), np.array([0.72 , 0.154, 0.73 , 0.693, 0.05 ]), np.array([0.057  , 0.49167, 0.24   , 0.218  , 0.42   , 0.73   ]), np.array([0.0564, 0.0659, 0.0229, 0.0387, 0.403 , 0.801 , 0.488 , 0.89  ])]   
            y1 = [-0.003762575048057693, 0.622924627471964, -0.038428939897216026, -4.049855859138972, 1091.1580176141754, -0.7226851853318268, 1.357713050588125, 9.59862285]
            
            X = np.array( [X1[idx-1]])    
            y.append(y1[(idx-1)])
         
        if add >= 2:  
            X2 = [np.array([0.651, 0.67 ]), np.array([0.7  , 0.826]), np.array([0.242114, 0.1     , 0.272433]),   np.array([0.57, 0.41, 0.43, 0.3 ]), np.array([0.32 , 0.746, 0.779, 0.78 ]), np.array([0.62, 0.15, 0.83, 0.8 , 0.03]), np.array([0.057  , 0.49167, 0.24, 0.218  , 0.42   , 0.73   ]), np.array([0.0564, 0.0659, 0.0229, 0.4   , 0.503 , 0.601 , 0.588 , 0.9   ])]
            y2 = [-0.0031077063415122937, 0.60224639848784, -0.1344876383977799, -3.754695676825303, 254.36133235195481, -0.6974609336778691, 1.357713050588125, 9.52782054]

            X = np.concatenate((X,[X2[idx-1]]),axis=0)
            y.append(y2[(idx-1)])

            
        # remember y is 1D    
        y =  np.array(y)  
        
        return X,y 

def print_for_capstone():
        X2  = [np.array([0.2, 0.3 ]), np.array([0.2  , 0.5]),      np.array([0.24211446, 0.64407427, 0.1]),  np.array([0.4, 0.3 ,0.6,0.3]),         np.array([0.4,0.4,0.4,0.4 ]), np.array([0.5,0.5,0.5,0.5,0.5]), np.array([0.1,0.1,0.1,0.1, 0.1]), np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])]
        for ar in X2:
            for a in ar:
                 print(f"{a:6f}-",end="")       
            print()
