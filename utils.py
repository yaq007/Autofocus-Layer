# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.initialized = False
#         self.val = None
#         self.avg = None
#         self.sum = None
#         self.count = None

#     def initialize(self, val, weight):
#         self.val = val
#         self.avg = val
#         self.sum = val * weight
#         self.count = weight
#         self.initialized = True

#     def update(self, val, weight=1):
#         if not self.initialized:
#             self.initialize(val, weight)
#         else:
#             self.add(val, weight)

#     def add(self, val, weight):
#         self.val = val
#         self.sum += val * weight
#         self.count += weight
#         self.avg = self.sum / self.count

#     def value(self):
#         return self.val

#     def average(self):
#         return self.avg

#     def summation(self):
#         return self.sum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count      