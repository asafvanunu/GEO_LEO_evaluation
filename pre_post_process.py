# %%
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd

# %%
def create_i_j(i_list, j_list):
    """this function get a list of i and a list of j and return them as one list of i,j

    Args:
        i_list (list): a list of fire pixels locations in row (i)
        j_list (list): a list of fire pixels locations in col (j)
    """
    out_list = []
    for i,j in zip(i_list, j_list):
        loc = (i, j)
        out_list.append(loc)
    return(out_list)

# %%
def out_of_bounds_test(location, matrix_shape, test_type):
    """get a location and a matrix shape and determine if it is out of bounds, and return nan or the location of the fire

    Args:
        location (list): location list for example [2,4] ([row,col])
        matrix_shape (tupple): matrix shape for example (10,10)
        test_type (string): write if it's for row or col. e.g. "row", "col"
    """
    min_row = 0 ## min row
    min_col = 0 ## min col
    max_row = matrix_shape[0] - 1 ## take the max row and col
    max_col = matrix_shape[1] - 1 
    loc_row = location[0] ## take the row and col location
    loc_col = location[1]
    
    if test_type == "row": ## if the test is for row
        if (loc_row < min_row) | (loc_row > max_row): ## if it is out of bounds 
            return(np.nan) ## return nan
        else:
            return(location) ## else return the location
    if test_type == "col":
        if (loc_col < min_col) | (loc_col > max_col):
            return(np.nan)
        else:
            return(location)

# %%
def set_R(location_row, location_col, R_type, matrix_shape, matrix):
    """this function get location of row and location of col and the R type we want

    Args:
        location_row (int): location of row
        location_col (int): location of col
        R_type (string): for example "R1" or "R2" "R3" "R4"
        matrix_shape(tupple): for example (10,10)
        matrix(2d array): The LEO rasterize array
    """
    if R_type == "R0":
        return(matrix[location_row, location_col])
    
    if R_type == "R1":
     ## If V(i,j+1)≤V(i,j) then R(1) = V(i,j+1)+ V(i,j)   
        R1_con = out_of_bounds_test(location=(location_row, location_col+1), matrix_shape=matrix_shape, test_type="col")
        if np.all(np.isnan(R1_con)):
            return(np.nan)
        else:
            R1_matrix = matrix[location_row, location_col+1]
            regular_value = matrix[location_row, location_col]
            if  R1_matrix <= regular_value:
                return(R1_matrix + regular_value)
            else:
                return(regular_value)
            
    if R_type == "R2":
     ## If V(i,j-1)≤V(i,j) then R(2) = V(i,j-1)+ V(i,j)   
        R2_con = out_of_bounds_test(location=(location_row, location_col-1), matrix_shape=matrix_shape, test_type="col")
        if np.all(np.isnan(R2_con)):
            return(np.nan)
        else:
            R2_matrix = matrix[location_row, location_col-1]
            regular_value = matrix[location_row, location_col]
            if  R2_matrix <= regular_value:
                return(R2_matrix + regular_value)
            else:
                return(regular_value)
            
    if R_type == "R3":
     ## If V(i+1,j)≤V(i,j) then R(3) = V(i+1,j)+ V(i,j)   
        R3_con = out_of_bounds_test(location=(location_row+1, location_col), matrix_shape=matrix_shape, test_type="row")
        if np.all(np.isnan(R3_con)):
            return(np.nan)
        else:
            R3_matrix = matrix[location_row+1, location_col]
            regular_value = matrix[location_row, location_col]
            if  R3_matrix <= regular_value:
                return(R3_matrix + regular_value)
            else:
                return(regular_value)
            
    if R_type == "R4":
     ## If V(i-1,j)≤V(i,j) then R(4) = V(i-1,j)+ V(i,j)   
        R4_con = out_of_bounds_test(location=(location_row-1, location_col), matrix_shape=matrix_shape, test_type="row")
        if np.all(np.isnan(R4_con)):
            return(np.nan)
        else:
            R4_matrix = matrix[location_row-1, location_col]
            regular_value = matrix[location_row, location_col]
            if  R4_matrix <= regular_value:
                return(R4_matrix + regular_value)
            else:
                return(regular_value)

# %%
def Max_R(location_row, location_col, matrix_shape, matrix):
    """this function get the set R function and return max R

    Args:
        location_row (int): location of row
        location_col (int): location of col
        R_type (string): for example "R1" or "R2" "R3" "R4"
        matrix_shape(tupple): for example (10,10)
        matrix(2d array): The LEO rasterize array
    """ 
    R0 = set_R(location_row=location_row, location_col=location_col, R_type="R0", matrix_shape=matrix_shape, matrix=matrix)
    R1 = set_R(location_row=location_row, location_col=location_col, R_type="R1", matrix_shape=matrix_shape, matrix=matrix)
    R2 = set_R(location_row=location_row, location_col=location_col, R_type="R2", matrix_shape=matrix_shape, matrix=matrix)
    R3 = set_R(location_row=location_row, location_col=location_col, R_type="R3", matrix_shape=matrix_shape, matrix=matrix)
    R4 = set_R(location_row=location_row, location_col=location_col, R_type="R4", matrix_shape=matrix_shape, matrix=matrix)
    
    R = [R0, R1, R2, R3, R4]
    return(np.nanmax(R))

# %%
def correct_LEO_matrix(original_matrix, threshold):
    """This function combines the functions above and produce an updated matrix

    Args:
        original_matrix (2d array): array
        threshold (int): an int for threshold
    """
    W = original_matrix.copy() # Set a copy of the original matrix
    matrix_shape = original_matrix.shape ## get the shape
    
    i_list, j_list = np.where(original_matrix>0) ## get where there are fire pixels
    
    i_j_list = create_i_j(i_list, j_list) ## create a list of these locations
    
    for loc in i_j_list: ## for each location
        loc_row = loc[0] ## take the row location
        loc_col = loc[1] ## take col location
        
        if W[loc_row, loc_col]<threshold: ## if the value is below the threshold
        ## calculate max R
            R = Max_R(location_row = loc_row, location_col = loc_col, matrix_shape = matrix_shape, matrix = original_matrix)
            W[loc_row, loc_col] = R ## apply the new value
        
    W[W<threshold] = 0 ## everything below the threshold set to zero
        
    return(W)

# %%
def sliding_window(arr:"array", window_size:"size of the window") -> "array sliced to the window":
    """ Construct a sliding window view of the array
    Keyword arguments:
   arr --  array, could be a raster image once it is open
   window_size --  for example "3" if we want a 3X3 window
    """
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

# %%
def cell_neighbors(arr:"array", i:"int number index", j:"int number index", d:"int number distance") -> "cell neighbors":
    """Return d-th neighbors of cell (i, j)
     Keyword arguments:
   arr --  array, could be a raster image once it is open
   i --  row index
   j --  column index
   d --  the distance from the pixel. for example "1" will act like a 3X3 window and "2" will be a distance of 2 from the index
    """
    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()

# %%
def find_FN(corrected_LEO_matrix, GEO_matrix, window_size, GEO_fire_value):
    """This function calculate the number of FN (false negative or omission) in a single GEO and LEO image. It returns a
        FN matrix where 1 means false negative pixel and zero means not false negative

    Args:
        corrected_LEO_matrix (array): corrected LEO matrix
        GEO_matrix (array): GEO matrix
        window_size (int): a number that is the size of the window for example: 0 for 1x1, 1 for 3X3, 2 for 5X5 and 3 for 7X7  
        GEO_fire_value (int\list): The GEO fire value in our example is 99 or [99,33]
    """
    
    FN = np.zeros((GEO_matrix.shape)) ## set an empty matrix with GEO matrix shape
    
    ## LEO fire locations
    LEO_fire_i, LEO_fire_j = np.where(corrected_LEO_matrix>0)
    ## For each LEO fire location:
    
    ## If the window size is 1X1 (no neighbors)
    if window_size == 0:
        for i_loc,j_loc in zip(LEO_fire_i, LEO_fire_j): ## for each LEO fire pixel
            GEO_value_in_LEO_loc = GEO_matrix[i_loc, j_loc] ## take GEO value in LEO fire location
            con_GEO_value = np.isin(GEO_value_in_LEO_loc, GEO_fire_value) ## if this value in the cell is not  fire 
            if con_GEO_value == False:
                FN[i_loc, j_loc] = 1 ## Mark it as false negative
        return(FN) ## Return the matrix
                
    else: ## If the window size is bigger than 1X1    
        for i_loc,j_loc in zip(LEO_fire_i, LEO_fire_j):
        
       ## take the GEO neighbors in a window from a LEO fire location
            LEO_GEO_fire_neighbor = cell_neighbors(arr=GEO_matrix, i=i_loc, j=j_loc, d=window_size)
        ## if there isn't a  GEO fire pixel near LEO fire location
            con_fire = np.any(np.isin(LEO_GEO_fire_neighbor, GEO_fire_value))
        
            if con_fire == False:
                FN[i_loc, j_loc] = 1 ## Mark it as false negative
            
        return(FN) ## Return the matrix

# %%
def out_of_bounds_test_value_return(location, matrix_shape, test_type, matrix):
    """get a location and a matrix shape and determine if it is out of bounds, and return nan or the value of the location

    Args:
        location (list): location list for example [2,4] ([row,col])
        matrix_shape (tupple): matrix shape for example (10,10)
        test_type (string): write if it's for row or col. e.g. "row", "col"
        matrix (array): The array we want to get the value
    """
    min_row = 0 ## min row
    min_col = 0 ## min col
    max_row = matrix_shape[0] - 1 ## take the max row and col
    max_col = matrix_shape[1] - 1 
    loc_row = location[0] ## take the row and col location
    loc_col = location[1]
    
    if test_type == "row": ## if the test is for row
        if (loc_row < min_row) | (loc_row > max_row): ## if it is out of bounds 
            return(np.nan) ## return nan
        else:
            return(matrix[loc_row, loc_col]) ## else return the location
    if test_type == "col":
        if (loc_col < min_col) | (loc_col > max_col):
            return(np.nan)
        else:
            return(matrix[loc_row, loc_col])

# %%
def post_FN(corrected_rasterized_LEO, FN_matrix, threshold):
    """This function calculate the number of FN (false negative or omission) in a single GEO and LEO image after pre-process. 
        It returns FN matrix where 1 means false negative pixel and zero means not false negative and also a df with FN summary

    Args:
        corrected_rasterized_LEO (array): corrected rasterized LEO array
        FN_matrix (array): pre-processed FN matrix
        threshold (int): threshold of LEO fire pixels inside a GEO pixel, for example 5
        
    """
    ## A new FN matrix in the size of the FN matrix
    ## FNnew = zeros(size(FN))
    FN_new = np.zeros(FN_matrix.shape)
    
    ## LEO fire locations
    ## For each location (i,j) not at the boundary
    LEO_fire_i, LEO_fire_j = np.where(corrected_rasterized_LEO>0)
    
    ## For each fire location
    for i_loc, j_loc in zip(LEO_fire_i, LEO_fire_j):
        
        ## If the value of the corrected LEO is higher than 2*threshold
        ## If W(i,j) >2t 
        if corrected_rasterized_LEO[i_loc, j_loc] > (2 * threshold):
            ## the new FN gets the value of the pre-process FN
            ## FNnew(i,j) = FN(i,j)
            FN_new[i_loc, j_loc] = FN_matrix[i_loc, j_loc]
         ## If the LEO fire value is equal or bigger from the t and also  smaller form 2*t 
         ## If W(i,,j) >= t and W(i,j) <= 2t  
        elif (corrected_rasterized_LEO[i_loc, j_loc] >= threshold) & (corrected_rasterized_LEO[i_loc, j_loc] <= (threshold * 2)):
            ## Cn = [W(i,j)==W(i+1,j),  W(i,j)==W(i-1,j), W(i,j)==W(i,j+1), W(i,j)==W(i,j-1)] 
            Cn = [] ## empty list for summing up neighbors
            W_i_j = corrected_rasterized_LEO[i_loc, j_loc] ## take the location of the LEO fire
            ## take the W(i+1,j)
            W_plus_row = out_of_bounds_test_value_return(location=[i_loc+1,j_loc], test_type="row",
                                            matrix_shape=FN_new.shape, matrix=corrected_rasterized_LEO)
            ## take the W(i-1,j)
            W_minus_row = out_of_bounds_test_value_return(location=[i_loc-1,j_loc], test_type="row",
                                            matrix_shape=FN_new.shape, matrix=corrected_rasterized_LEO)
            ## take the W(i,j+1)
            W_plus_col = out_of_bounds_test_value_return(location=[i_loc,j_loc+1], test_type="col",
                                            matrix_shape=FN_new.shape, matrix=corrected_rasterized_LEO)
            ## take the W(i,j-1)
            W_minus_col = out_of_bounds_test_value_return(location=[i_loc,j_loc-1], test_type="col",
                                            matrix_shape=FN_new.shape, matrix=corrected_rasterized_LEO)
            ## Append all
            if W_i_j == W_plus_row:
                Cn.append(1)
            else:
                Cn.append(0)
            if W_i_j == W_minus_row:
                Cn.append(1)
            else:
                Cn.append(0)
            if W_i_j == W_plus_col:
                Cn.append(1)
            else:
                Cn.append(0)
            if W_i_j == W_minus_col:
                Cn.append(1)
            else:
                Cn.append(0)
            
            #Fneighbor = [FN(i+1,j), FN(i-1,j), FN(i,j+1), FN(i,j-1)]
            Fneighbor = []
            ## take the F(i+1,j)
            F_plus_row = out_of_bounds_test_value_return(location=[i_loc+1,j_loc], test_type="row",
                                            matrix_shape=FN_new.shape, matrix=FN_matrix)
            ## take the F(i-1,j)
            F_minus_row = out_of_bounds_test_value_return(location=[i_loc-1,j_loc], test_type="row",
                                            matrix_shape=FN_new.shape, matrix=FN_matrix)
            ## take the F(i,j+1)
            F_plus_col = out_of_bounds_test_value_return(location=[i_loc,j_loc+1], test_type="col",
                                            matrix_shape=FN_new.shape, matrix=FN_matrix)
            ## take the F(i,j-1)
            F_minus_col = out_of_bounds_test_value_return(location=[i_loc,j_loc-1], test_type="col",
                                            matrix_shape=FN_new.shape, matrix=FN_matrix)
            
            Fneighbor.append(F_plus_row)
            Fneighbor.append(F_minus_row)
            Fneighbor.append(F_plus_col)
            Fneighbor.append(F_minus_col)
            ## If sum(Cn) = 0  
            if sum(Cn) == 0:
                FN_new[i_loc, j_loc] = FN_matrix[i_loc, j_loc]
            
            else:
                if FN_matrix[i_loc, j_loc] == 1:
                    ## FNnew(i,j) = ½ * sum( Cn * Fneighbor)
                    Cn_array = np.array(Cn)
                    Fneighbor_array = np.array(Fneighbor) 
                    FN_new[i_loc, j_loc] = 0.5 * np.nansum(Cn_array * Fneighbor_array)
                    
    return(FN_new)

# %%
def false_alarm_dummy_rast_non_max_sup(dummy_rast):
    """Gets a dummy rasterized file and return a false alarm gdf
    Keyword arguments

    Args:
        dummy_rast (array): raster dummy rast
    """
    n = np.sum(np.isnan(dummy_rast) == False) ## total number of pixels that are not nan
    
    ## Calcualte statistics
    
    ON = np.sum(dummy_rast == 0) ## Omission no fire - True positive
    
    OF =  np.sum(dummy_rast[(dummy_rast<88) & (dummy_rast>0)]) ## Omission fire - False negative
    
    CN = np.sum(dummy_rast == 88) ## Commsion no fire - False positve
    
    CF = np.sum(dummy_rast == 99) ## Comission fire - True positive
    
    
    # create dict
    d = {"number_of_pixels_(n)":[n], "True positive":[ON],
        "False negative":[OF], "False positive":[CN], "True positive":[CF]}
    
    # create df
    df = pd.DataFrame(data=d)
    
    return(df)

# %%
np.zeros((3,3))

# %%
def calculate_false_alarm_GEO_LEO(corrected_rasterized_LEO_matrix, distance_buffer, GEO_matrix,
                                      fire_label_GEO, threshold, return_image):
    """Gets a rasterized file and return a false alarm df 

    Args:
        corrected_rasterized_LEO (array): rasterized LEO array after correction
        distance_buffer (int): the distance of the buffer. e.g 0 is 1x1 (no buffer) 1 is 3x3 buffer, 2 is 5x5 buffer, 3 is 7x7
        GEO_matrix (array): The GEO array
        fire_label_GEO (int\list): the fire label in a GEO matrix. e.g 99 or [99, 33]
    
        threshold (int): Number of LEO pixels threshold inside a GEO matrix
        return_image (string): "Y" to return and "N" to return only df
    """
    ## In case we calculate without a buffer:
    
    if distance_buffer == 0:
        
        
        dummy_rast = np.zeros(GEO_matrix.shape) ## set a zero array in the shape of the GEO matrix
            
            
            ## Get a False negative matrix
            
        FN_matrix = find_FN(corrected_LEO_matrix=corrected_rasterized_LEO_matrix, GEO_matrix=GEO_matrix,
                            window_size=distance_buffer, GEO_fire_value=fire_label_GEO)
        
            ## correct it
            
        FN_corrected_matrix = post_FN(corrected_rasterized_LEO=corrected_rasterized_LEO_matrix, FN_matrix=FN_matrix,
                                      threshold=threshold)
        
            ## Take all the values of the FN corrected matrix and insert them to the dummy rast
        dummy_rast[FN_corrected_matrix != 0] = FN_corrected_matrix[FN_corrected_matrix!=0]
            
            ### GEO matrix PART ########
        
        GEO_row_i, GEO_col_j = np.where(np.isin(GEO_matrix,fire_label_GEO)) ## take the row and col index for all GEO pixels that are fire
        ## in a given pixel label
        
        for i, j in zip(GEO_row_i, GEO_col_j): ## for row index i and col index j
                
                ## take the LEO value in GEO fire locations
            LEO_value = corrected_rasterized_LEO_matrix[i,j]
                
            if (LEO_value>0) == True: ## if the LEO cell value is heigher than 0
                dummy_rast[i,j] = 99 ## set this pixel as fire (99 means fire)
            else: ## if LEO cells dont have fire
                dummy_rast[i,j] = 88 ## set it to 88 - false alarm!
        
        ## calculate false alarm
        df_no_neighbor = false_alarm_dummy_rast_non_max_sup(dummy_rast=dummy_rast)
        
        dummy_rast[np.isnan(GEO_matrix)] = np.nan ## set nan values according to GEO matrix
                
        if (return_image == "Y"): ## in case we want to also see the image to analyze it!
            return(df_no_neighbor, dummy_rast)
        if (return_image == "N"):
            return(df_no_neighbor)
        
        ## If there is a buffer in the calculations:
    else:
          
        dummy_rast = np.zeros(GEO_matrix.shape) ## set a zero array in the shape of the GEO matrix.copy()
            
        ## Get the false negative matrix    
        FN_matrix = find_FN(corrected_LEO_matrix=corrected_rasterized_LEO_matrix, GEO_matrix=GEO_matrix,
                            window_size=distance_buffer, GEO_fire_value=fire_label_GEO)
        
            ## correct it
            
        FN_corrected_matrix = post_FN(corrected_rasterized_LEO=corrected_rasterized_LEO_matrix, FN_matrix=FN_matrix,
                                      threshold=threshold)
        
            ## Take all the values of the FN corrected matrix and insert them to the dummy rast
        dummy_rast[FN_corrected_matrix != 0] = FN_corrected_matrix[FN_corrected_matrix!=0]
            
            ### GEO matrix PART ########
        
        GEO_row_i, GEO_col_j = np.where(np.isin(GEO_matrix,fire_label_GEO)) ## take the row and col index for all GEO pixels that are fire
        
        for i, j in zip(GEO_row_i, GEO_col_j): ## for row index i and col index j
                
                ## take the LEO neighbors in the given distance
            neighboors = cell_neighbors(arr=corrected_rasterized_LEO_matrix, i = i, j= j, d=distance_buffer) 
                
                ## if all the neighbors are equal to zero
            calc = np.all(neighboors == 0)
                
            if (np.any(neighboors>0)) == True: ## if any of the neighbors value is heigher than 0
                dummy_rast[i,j] = 99 ## set this pixel as fire (99 means fire)
            if calc == True: ## if all of the LEO cells srounding neighbors dont have fire
                dummy_rast[i,j] = 88 ## set it to 88 - false alarm!
        
        ## calculate false alarm
        df = false_alarm_dummy_rast_non_max_sup(dummy_rast=dummy_rast)
        
        dummy_rast[np.isnan(GEO_matrix)] = np.nan ## set nan values according to GEO matrix
                
        if (return_image == "Y"): ## in case we want to also see the image to analyze it!
            return(df, dummy_rast)
        if (return_image == "N"):
            return(df)

# %%



