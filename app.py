from glob import glob
import numpy as np
import os
from main import Network
from main import utls
import time
import cv2
import keras
from flask import Flask, request
from werkzeug.utils import secure_filename

model_name = 'Syn_img_lowlight_withnoise'
mbllen = Network.build_mbllen((None, None, 3))
mbllen.load_weights('./models/'+model_name+'.h5')
opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
mbllen.compile(loss='mse', optimizer=opt)

def update_image(img_path):
   flag = 1
   lowpercent = 5
   highpercent = 95
   maxrange = 8/10.
   hsvgamma = 8/10.

   img_A_path = img_path
   img_A = utls.imread_color(img_A_path)
   img_A = img_A[np.newaxis, :]

   starttime = time.process_time()
   out_pred = mbllen.predict(img_A)
   endtime = time.process_time()
   print('The image\'s Time:' +str(endtime-starttime)+'s.')
   fake_B = out_pred[0, :, :, :3]
   fake_B_o = fake_B

   gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
   percent_max = sum(sum(gray_fake_B >= maxrange))/sum(sum(gray_fake_B <= 1.0))
   # print(percent_max)
   max_value = np.percentile(gray_fake_B[:], highpercent)
   if percent_max < (100-highpercent)/100.:
      scale = maxrange / max_value
      fake_B = fake_B * scale
      fake_B = np.minimum(fake_B, 1.0)

   gray_fake_B = fake_B[:,:,0]*0.299 + fake_B[:,:,1]*0.587 + fake_B[:,:,1]*0.114
   sub_value = np.percentile(gray_fake_B[:], lowpercent)
   fake_B = (fake_B - sub_value)*(1./(1-sub_value))

   imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
   H, S, V = cv2.split(imgHSV)
   S = np.power(S, hsvgamma)
   imgHSV = cv2.merge([H, S, V])
   fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
   fake_B = np.minimum(fake_B, 1.0)

   if flag:
      outputs = np.concatenate([img_A[0,:,:,:], fake_B_o, fake_B], axis=1)
   else:
      outputs = fake_B

   img_name = os.path.basename(img_path)
   # scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(img_name)
   outputs = np.minimum(outputs, 1.0)
   outputs = np.maximum(outputs, 0.0)
   utls.imwrite(img_name, outputs)

app = Flask(
    __name__,
    static_url_path='', 
    static_folder=''
)

def get_form():
    return """
        <form action="/" enctype="multipart/form-data" method="POST">
            <input type="file" id="file" name="file">
            <input type="submit">
        </form>            
    """


@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return get_form()
    elif request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        f.save(secure_filename(f.filename))
        update_image(f.filename)
        return f'{get_form()} <img src="{f.filename}"/>'


if __name__ == "__main__":
    app.run("0.0.0.0")