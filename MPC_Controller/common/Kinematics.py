def inv_kine_hip(tip_pos,bend_info):
    """inverse kinematics in leg hip frame
        tip_pos: ndarray (6,3)
        bend_info: ndarray (6,)
    """
    # t1 > 0 , t2 < 0 for bend in. bend_info=1
    # t1 < 0 , t2 > 0 for bend out. bend_info=0
    joint_pos = np.zeros((6,3))
    for i in range(6):
        x = tip_pos[i,0]
        y = tip_pos[i,1]
        z = tip_pos[i,2]
        theta0 = math.atan(-y/z)

        z_tip_sag = math.sqrt(z*z+y*y)
        cos_shank = (z_tip_sag**2 + x**2 - len_thigh**2 - len_calf**2)/(2*len_thigh*len_calf)
        if cos_shank>1:
            cos_shank = 1
        if cos_shank<-1:
            cos_shank = -1

        if bend_info[i] == 1:
            theta2 = - math.acos(cos_shank)
        else:
            theta2 = math.acos(cos_shank)

        cos_beta = (z_tip_sag**2 + x**2 + len_thigh**2 - len_calf**2)/(2*len_thigh*math.sqrt(z_tip_sag**2 + x**2))
        if cos_beta>1:
            cos_beta = 1
        if cos_beta<-1:
            cos_beta = -1
        beta = math.acos(cos_beta)

        alpha = math.atan(x/z_tip_sag)
        if x>0:
            if bend_info[i] == 0:
                theta1 = -alpha - beta
            else:
                theta1 = -alpha + beta
        else:
            if bend_info[i] == 0:
                theta1 = -alpha - beta
            else:
                theta1 = -alpha + beta

        joint_pos[i,0] = theta0
        joint_pos[i,1] = theta1
        joint_pos[i,2] = theta2

    return joint_pos