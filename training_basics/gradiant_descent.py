def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradiant(x,y,w,b):
    m = x.shape[0]
    total_w =0
    total_b = 0
    
    for i in range(m):
        f_wb = w* x[i] + b
        dj_dw = (f_wb - y[i]) * x[i]
        dj_db = (f_wb - y[i])
        total_w += dj_dw
        total_b += dj_db
    total_w = total_w/m
    total_b = total_b/m
    
    
    return total_w,total_b 