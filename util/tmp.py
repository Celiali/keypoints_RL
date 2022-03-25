if r is True:
    rightmiss = np.array(np.where(np.linalg.norm(rightmarker, axis=1) < 0.0001))
    rightexist = np.array(np.where(np.linalg.norm(rightmarker, axis=1) >= 0.0001))
    rightspeed = (rightmarker[rightexist.max()] - rightmarker[rightexist.min()]) / (rightexist.max() - rightexist.min())

    for i_right in rightmiss:
        if i_right < rightexist.min():
            step = rightexist.min() - i_right
            rightmarker[i_right] = rightmarker[rightexist.min()] - rightspeed * step
            continue
        if i_right > rightexist.max():
            step = i_right - rightexist.min()
            rightmarker[i_right] = rightmarker[rightexist.max()] + rightspeed * step
            continue

        rightmarker[i_right] = rightmarker[i_right - 1] + rightspeed