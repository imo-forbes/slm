# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:18:33 2021

@author: imoge
"""
import numpy as np
import matplotlib.pyplot as plt

#data for z2,0 (-100,100,5)

x_max_int_1 = np.array([5259.0, 5212.0, 5233.0, 5053.0, 5007.0, 4885.0, 4759.0, 4757.0, 4697.0, 4634.0, 4567.0, 4515.0, 4357.0, 4200.0, 3962.0, 3793.0, 3673.0, 3676.0, 3869.0, 5192.0, 5527.0, 4795.0, 3864.0, 3636.0, 3648.0, 3731.0, 3893.0, 4127.0, 4231.0, 4338.0, 4523.0, 4673.0, 4721.0, 4782.0, 4832.0, 4907.0, 5048.0, 5206.0, 5251.0, 5370.0])
y_max_int_1 = np.array([5788.0, 5676.0, 5544.0, 5381.0, 5362.0, 5321.0, 5121.0, 5078.0, 5166.0, 5200.0, 5084.0, 4997.0, 4810.0, 4794.0, 4501.0, 4483.0, 4690.0, 4540.0, 4315.0, 5802.0, 6092.0, 5551.0, 4185.0, 4395.0, 4560.0, 4619.0, 4660.0, 5020.0, 5217.0, 5260.0, 5385.0, 5309.0, 5410.0, 5512.0, 5479.0, 5695.0, 5763.0, 5813.0, 5817.0, 5901.0])
x_max_int_2 = np.array([5402.0, 5348.0, 5349.0, 5211.0, 5179.0, 5027.0, 4933.0, 4941.0, 4935.0, 4882.0, 4813.0, 4698.0, 4580.0, 4430.0, 4159.0, 3996.0, 3887.0, 3928.0, 4125.0, 5343.0, 5628.0, 5013.0, 4051.0, 3844.0, 3874.0, 3972.0, 4134.0, 4311.0, 4435.0, 4555.0, 4777.0, 4864.0, 4949.0, 5039.0, 5095.0, 5154.0, 5226.0, 5396.0, 5439.0, 5529.0])
y_max_int_2 = np.array([5926.0, 5788.0, 5770.0, 5704.0, 5521.0, 5559.0, 5522.0, 5481.0, 5366.0, 5432.0, 5438.0, 5361.0, 5201.0, 5123.0, 4986.0, 4599.0, 4620.0, 4517.0, 4824.0, 6143.0, 5989.0, 5424.0, 4750.0, 4633.0, 4489.0, 4742.0, 5008.0, 5207.0, 5203.0, 5277.0, 5508.0, 5629.0, 5665.0, 5475.0, 5814.0, 5605.0, 5687.0, 5874.0, 5840.0, 6089.0])
x_max_int_3 = np.array([5383.0, 5383.0, 5340.0, 5186.0, 5200.0, 5087.0, 5005.0, 4976.0, 4883.0, 4868.0, 4715.0, 4604.0, 4396.0, 4243.0, 4069.0, 3893.0, 3874.0, 3949.0, 4301.0, 5628.0, 5641.0, 5103.0, 4124.0, 3870.0, 3865.0, 3968.0, 4192.0, 4402.0, 4481.0, 4653.0, 4821.0, 4928.0, 5000.0, 5109.0, 5183.0, 5202.0, 5339.0, 5481.0, 5536.0, 5656.0])
y_max_int_3 = np.array([5966.0, 6212.0, 5878.0, 5734.0, 5572.0, 5594.0, 5495.0, 5422.0, 5629.0, 5555.0, 5549.0, 5419.0, 5236.0, 5270.0, 4871.0, 4607.0, 4675.0, 4625.0, 4733.0, 6088.0, 6232.0, 5801.0, 4769.0, 4567.0, 4594.0, 4687.0, 5071.0, 5178.0, 5430.0, 5410.0, 5771.0, 5738.0, 5626.0, 5772.0, 5909.0, 5771.0, 5756.0, 5978.0, 6130.0, 6192.0])


#data for z4,4 (-100,100,5)
x_max_int_1 = np.array([5805.0, 5822.0, 5878.0, 5981.0, 6046.0, 6082.0, 6182.0, 6322.0, 6480.0, 6610.0, 6772.0, 6911.0, 7112.0, 7284.0, 7472.0, 7531.0, 7654.0, 7828.0, 8072.0, 8697.0, 5516.0, 9157.0, 8555.0, 8346.0, 8166.0, 8078.0, 7952.0, 7823.0, 7624.0, 7471.0, 7293.0, 7122.0, 7027.0, 6785.0, 6668.0, 6541.0, 6427.0, 6405.0, 6296.0, 6264.0])
y_max_int_1 = np.array([6476.0, 6370.0, 6524.0, 6539.0, 6573.0, 6776.0, 6912.0, 7149.0, 7285.0, 7406.0, 7717.0, 7782.0, 8063.0, 8296.0, 8348.0, 8426.0, 8491.0, 8717.0, 9160.0, 9443.0, 6067.0, 8296.0, 7957.0, 8123.0, 8271.0, 8333.0, 8272.0, 8325.0, 8043.0, 7812.0, 7654.0, 7844.0, 7350.0, 7325.0, 7045.0, 6920.0, 6866.0, 6725.0, 6840.0, 6611.0])
x_max_int_2 = np.array([6175.0, 6217.0, 6278.0, 6347.0, 6447.0, 6553.0, 6651.0, 6728.0, 6852.0, 6920.0, 7064.0, 7203.0, 7308.0, 7365.0, 7526.0, 7633.0, 7665.0, 7891.0, 8260.0, 8855.0, 5626.0, 9100.0, 8387.0, 8199.0, 8145.0, 8044.0, 7903.0, 7800.0, 7667.0, 7567.0, 7443.0, 7304.0, 7148.0, 7044.0, 6875.0, 6831.0, 6666.0, 6665.0, 6510.0, 6502.0])
y_max_int_2 = np.array([6745.0, 6734.0, 7024.0, 7004.0, 6959.0, 7212.0, 7152.0, 7483.0, 7580.0, 7848.0, 7839.0, 7913.0, 8286.0, 8178.0, 8539.0, 8503.0, 8923.0, 8956.0, 9464.0, 9720.0, 6095.0, 8459.0, 8075.0, 8157.0, 8280.0, 8302.0, 8148.0, 8258.0, 7971.0, 7979.0, 7862.0, 7586.0, 7512.0, 7356.0, 7208.0, 7001.0, 6946.0, 7037.0, 6793.0, 6810.0])
x_max_int_3 = np.array([5854.0, 5897.0, 5943.0, 6035.0, 6212.0, 6307.0, 6522.0, 6691.0, 6858.0, 6995.0, 7162.0, 7381.0, 7541.0, 7695.0, 7841.0, 7929.0, 8018.0, 8198.0, 8335.0, 8544.0, 5654.0, 9152.0, 8556.0, 8280.0, 8148.0, 8044.0, 7972.0, 7806.0, 7719.0, 7550.0, 7357.0, 7175.0, 6995.0, 6866.0, 6747.0, 6653.0, 6509.0, 6547.0, 6333.0, 6325.0])
y_max_int_3 = np.array([6418.0, 6689.0, 6578.0, 6818.0, 6907.0, 7052.0, 7401.0, 7326.0, 7568.0, 7975.0, 7953.0, 8234.0, 8557.0, 8697.0, 8679.0, 8898.0, 8966.0, 9182.0, 9472.0, 9688.0, 6057.0, 8285.0, 8132.0, 8348.0, 8247.0, 8154.0, 8250.0, 8195.0, 7946.0, 7850.0, 7821.0, 7470.0, 7301.0, 7420.0, 7010.0, 6823.0, 6848.0, 6907.0, 6754.0, 6758.0])
z=np.arange(-100,100,5)



print(len(x_max_int_1))

plt.figure()
plt.scatter(z, x_max_int_1, label ='Measurement 1')
plt.scatter(z, x_max_int_2, label = 'Measurement 2')
plt.scatter(z, x_max_int_3, label = 'Measurement 3')
plt.xlabel('Amplitude')
plt.ylabel('Intensity in x')
plt.legend()

plt.figure()
plt.scatter(z, y_max_int_1, label ='Measurement 1')
plt.scatter(z, y_max_int_2, label = 'Measurement 2')
plt.scatter(z, y_max_int_3, label = 'Measurement 3')
plt.xlabel('Amplitude')
plt.ylabel('Intensity in y')
plt.legend()

plt.figure()
plt.scatter(z, ((x_max_int_1 + x_max_int_2 + x_max_int_3)/3), label = 'Mean Intensity in x')
plt.scatter(z, ((y_max_int_1+y_max_int_2+ y_max_int_3)/3), label = 'Mean Intensity in y')
plt.xlabel('Amplitude')
plt.ylabel('Intensity')
plt.legend(loc=2, prop={'size': 8})

i_x = ((x_max_int_1 + x_max_int_2 + x_max_int_3)/3)
i_y = ((y_max_int_1+y_max_int_2+ y_max_int_3)/3)
print(i_x.max())
print(i_y.max())
z_x = np.where(i_x == i_x.max())
z_y = np.where(i_y == i_y.max())
print(z_x)
print(z_y)
#print(z[])
print(z[19])
            