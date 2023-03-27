# Distributed-Multiple-Model-Optimization
本脚本解决的问题是，对已经坏掉的分布式多节点权重矩阵，通过部分自定义梯度，优化模型，使其优于单节点和错误的配置矩阵的多节点模型
已经坏掉的分布式多节点权重矩阵：
        
        
        在configs.py里的ar，ac分别对应主模型和辅助模型


具体数学过程在main.py里的 update_grad 和update_xy-grad
