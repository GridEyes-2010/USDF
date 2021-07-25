# USDF
unsigned distance field 
该算法使用了快速壁面距离fcpw库，矩阵库Eigen，并行库ppl。目前该算法只使用了CPU进行计算。效率基本满足要求。后续进一步会对远场进行优化，预计会实现SIMD版本和ppl并行版本进行计算
