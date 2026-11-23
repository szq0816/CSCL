import numpy as np


def cal_std(logger, *arg):
    """ print clustering results """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.4f} std {:.4f}
                     NMI {:.4f} std {:.4f} 
                     ARI {:.4f} std {:.4f}""".format(np.mean(arg[0]) , np.std(arg[0]) , np.mean(arg[1]) ,
                                                     np.std(arg[1]) , np.mean(arg[2]) , np.std(arg[2]) )
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) , 2)) + ',' + str(round(np.std(arg[0]) , 2)) + ';' + \
                  str(round(np.mean(arg[1]) , 2)) + ',' + str(round(np.std(arg[1]) , 2)) + ';' + \
                  str(round(np.mean(arg[2]) , 2)) + ',' + str(round(np.std(arg[2]) , 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) , 2), round(np.mean(arg[1]) , 2), round(np.mean(arg[2]) , 2)

    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.4f} std {:.4f}""".format(np.mean(arg) , np.std(arg) )
        logger.info(output)


def cal_HAR(logger, *arg):
    """ print classification results for HAR """
    if len(arg) == 5:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        logger.info(arg[3])
        logger.info(arg[4])
        output = """ 
                     RGB {:.4f} std {:.4f}
                     Depth {:.4f} std {:.4f} 
                     RGB+D {:.4f} std {:.4f}
                     onlyrgb {:.4f} std {:.4f}
                     onlydepth {:.4f} std {:.4f}""".format(np.mean(arg[0]) , np.std(arg[0]) ,
                                                           np.mean(arg[1]) , np.std(arg[1]) ,
                                                           np.mean(arg[2]) , np.std(arg[2]) ,
                                                           np.mean(arg[3]) , np.std(arg[3]) ,
                                                           np.mean(arg[4]) , np.std(arg[4]) )

        logger.info(output)
    return


def cal_classify(logger, *arg):
    """ print classification results """
    if len(arg) == 3:
        logger.info(arg[0])
        logger.info(arg[1])
        logger.info(arg[2])
        output = """ 
                     ACC {:.4f} std {:.4f}
                     Precision {:.4f} std {:.2f} 
                     F-measure {:.4f} std {:.2f}""".format(np.mean(arg[0]) , np.std(arg[0]) ,
                                                           np.mean(arg[1]) ,
                                                           np.std(arg[1]) , np.mean(arg[2]) ,
                                                           np.std(arg[2]) )
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) , 2)) + ',' + str(round(np.std(arg[0]) , 2)) + ';' + \
                  str(round(np.mean(arg[1]) , 2)) + ',' + str(round(np.std(arg[1]) , 2)) + ';' + \
                  str(round(np.mean(arg[2]) , 2)) + ',' + str(round(np.std(arg[2]) , 2)) + ';'
        logger.info(output2)
        return round(np.mean(arg[0]) , 2), round(np.mean(arg[1]) , 2), round(np.mean(arg[2]) , 2)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.4f} std {:.4f}""".format(np.mean(arg) , np.std(arg) )
        logger.info(output)
    return
