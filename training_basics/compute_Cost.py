m = x.shape[0]


cost_sum =0.0


for i in range(m):
    f_wb = (w*x[i]) + b
    j_wb = (f_wb - y[i]) ** 2
    cost_sum+=j_wb

total_cost = cost_sum * (1/(2*m))

return total_cost