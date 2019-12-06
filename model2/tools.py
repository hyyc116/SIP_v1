#coding:utf-8
'''
保存一些基本的方法
'''


def is_better_result(mae,mse,r2,best_mae,best_mse,best_r2,best_score):

    if r2/(mae+mse)>=best_score*0.97:

        if mae<best_mae or mse<best_mse or r2>best_r2:
            return True
    
    return False