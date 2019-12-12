#coding:utf-8
'''
保存一些基本的方法
'''


def is_better_result(mse,best_mse):

    if mse<best_mse:
        return True
    
    return False