import torch
import torch.nn as nn
from fastai.basic_train import Learner, load_learner
from fastai.vision import ImageDataBunch,ImageList
from fastai.basic_data import DataBunch
from torchvision.models.video import r3d_18, r2plus1d_18, mc3_18
from fastai.layers import bn_drop_lin
from fastai.vision import Image as FastaiImage
import os
import shutil
import fnmatch
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from torchvision.io import read_video
import enum
from pytube import YouTube
from time import gmtime, strftime
import streamlit as st
import tempfile
import subprocess
from streamlit_player import st_player

try:
    import streamlit.ReportThread as ReportThread
    from streamlit.server.Server import Server
except Exception:
    # Streamlit >= 0.65.0
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server


class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def get(**kwargs):
    ctx = ReportThread.get_report_ctx()

    this_session = None

    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        # Streamlit < 0.56
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (
            # Streamlit < 0.54.0
            (hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg)
            or
            # Streamlit >= 0.54.0
            (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue)
            or
            # Streamlit >= 0.65.2
            (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr)
        ):
            this_session = s

    if this_session is None:
        raise RuntimeError(
            "Oh noes. Couldn't get your Streamlit Session object. "
            'Are you doing something fancy with threads?')

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state





IMG_SIZE = (112,112)

class Models(enum.Enum):
    R2PLUS1D_18 = 'r2plus1d_18'
    R3D_18 = 'r3d_18'
    R2PLUS1D_50 = 'r2plus1d_50_KH'
    MC3_18 = 'mc3_18'

from google_drive_downloader import GoogleDriveDownloader as gdd


URLs = {Models.R3D_18:'1KWaHxwxCY8izU7rLA1DhTjlbfQi8sPDh'}

class Data:
    def __init__(self):
        self.path = '.'
        self.device = torch.device('cpu')
        self.loss_func = None





def load_mc3_model(fn,two_layer_head=False):
    model = create_base_mc3_model()
    learner = Learner(Data(),model)
    learner.model = model
    learner.load(fn)
    return learner.model



def create_2layers_head(input_size=512, drop_out_p=0):
    b1 = bn_drop_lin(input_size, input_size, True, drop_out_p, nn.ReLU())
    b2 = bn_drop_lin(input_size, 2, True, drop_out_p, nn.ReLU())
    layers = []
    layers += b1
    layers += b2
    head = nn.Sequential(*layers)
    # head.add_module(b1)
    # head.add_module(b2)
    return head


def create_base_mc3_model(two_layer_head=False, drop_out_p=0):
    model = mc3_18(pretrained=False)
    model.fc = create_2layers_head(drop_out_p=drop_out_p) if two_layer_head else nn.Linear(512, 2)

    model.name = 'mc3'
    print(model)
    return model


class Normalize4d:
    def __init__(self, mean=[0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989], inplace=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1, 1)
        tensor.sub_(mean).div_(std)
        return tensor



class VolumeImage(FastaiImage):
    def __init__(self, px):
        super().__init__(px)

    def apply_tfms(self, *args, **kwargs):
        # print("******Runing apply tfms*****")
        '''
        for i  in range(self._px.shape[0]):
            data = self._px[i,::,::]
            data = data.unsqueeze(0)
            img = Image(data)
            img.apply_tfms(*args,**kwargs)
            self._px[i, ::, ::] = img.data[0,::,::]
        '''
        return self


class VolumeImageList(ImageList):
    # if normalize is defined fastai try to call it expecting normalize to be 3 values array
    def normalize_(self, data):
        self.nz(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # super().__init__()
        self.c, self.sizes = 2, {}
        # self.crop = True
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        self.nz = Normalize4d(mean, std, inplace=True)

    def open(self, fn):
        CROP_Y = 121 - 112
        CROP_X = 171 - 112
        data = torch.load(fn)
        size = data.size()
        '''
        if size[3] != 112:
            row = random.randint(0, CROP_Y - 1)
            col = random.randint(0, CROP_X - 1)
            # print("Data size before Crop:",data.size(),"row",row,"col",col)
            data = data[:, :, row:row + 112, col:col + 112]
            # print("Data size after Crop:",data.size())
        '''
        self.normalize_(data)
        return VolumeImage(data)

def get_image_4d_tensor(fn, img_size=IMG_SIZE):
    img = Image.open(fn)
    img = img.convert('RGB')
    img = img.resize(img_size)
    px = ToTensor()(img)
    px.unsqueeze_(1)
    return px




def get_image_fn_file_pattern(frame_number, file_pattern="frame_{:06d}.png"):
    return file_pattern.format(frame_number)


def get_3d_volume_file_pattetn(image_dir, start_frame, num_channels, file_pattern="frame_{:06d}.png",
                               img_size=IMG_SIZE):
    volume = []
    for i_frame in range(0, num_channels):
        # horizontal components
        frame_number = start_frame + i_frame
        # XXX: some tif images fail to open could be related to  https://github.com/python-pillow/Pillow/issues/4237
        try:
            fn = os.path.join(image_dir, get_image_fn_file_pattern(frame_number, file_pattern))
            # print(fn)
            px = get_image_4d_tensor(fn, img_size)
            volume.append(px)
        except:
            print("Could not open:", fn)

    result = torch.cat(volume, dim=1)
    return result

def get_number_of_files_in_folder(folder, file_pattern='*.png'):
    pattern = "*."+file_pattern.split('.')[1]
    return len(fnmatch.filter(os.listdir(folder), pattern))


def create_base_volumes(input_dir,output_dir,num_channels,file_pattern="frame_{:06d}.png"):
    num_files = get_number_of_files_in_folder(input_dir,file_pattern=file_pattern)

    print('num files', num_files, 'file pattern',file_pattern)
    start_frame = 1

    while start_frame + num_channels <= num_files:
        data = get_3d_volume_file_pattetn(input_dir, start_frame, num_channels,file_pattern=file_pattern,
                                          img_size=(112, 112))
        file_name = "volume_{:06d}.png".format( start_frame)
        file = os.path.join(output_dir, file_name)
        torch.save(data, file)
        start_frame += num_channels



class InferenceConfig:
    def __init__(self):
        base_tmp_dir = tempfile.gettempdir()
        self.num_channels = 16
        self.model_name = Models.MC3_18
        self.input_dir = os.path.join(base_tmp_dir,"frames")
        self.volume_dir = os.path.join(base_tmp_dir,"volumes")
        self.clean_volume_dir = True
        self.model_fn = "mc3_best_model_acc.pth"
        self.file_pattern = "frame_{:06d}.png"
        self.i2c={0:'Normal',1:'Abnormal'}
        self.download_dir = os.path.join(base_tmp_dir,"downloads")
        #self.ffmpeg_bin_path = "C:\\java\\ffmpeg-4.3.2-2021-02-27-essentials_build\\bin\\ffmpeg.exe"
        self.ffmpeg_bin_path = "ffmpeg"
        self.model_dir = os.path.join(base_tmp_dir,"models")
        self.download_model_from_google_drive = True
        self.frame_width = 400

class ProgressLogger:

    def __init__(self,st_logger):
        self.st_logger = st_logger
        self.logged_messages = ''
    def log(self,message):
        if self.st_logger is not None:
            self.logged_messages += message +'\n'
            self.st_logger.text(self.logged_messages)


def run_command(args):
    """Run command, transfer stdout/stderr back into Streamlit and manage error"""
    #st.info(f"Running '{' '.join(args)}'")
    result = subprocess.run(args, capture_output=True, text=True)
    try:
        result.check_returncode()
        #st.info(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        st.error(result.stderr)
        raise e

def get_frame_width(input_dir):
    image_fns = fnmatch.filter(os.listdir(input_dir), '*.png')
    if len(image_fns) > 0 :
        img = Image.open(os.path.join(input_dir,image_fns[0]))
        width,height = img.size
        return width
    return 400


class YoutubeVolumeCreator:
    def __init__(self,inferenceCondfig:InferenceConfig,progressLogger:ProgressLogger):
        self.inferenceConfig = inferenceCondfig
        self.progressLogger = progressLogger

    def download(self):
        url = self.inferenceConfig.youtube_url
        self.progressLogger.log("Downloading URL:"+url)
        download_dir = self.inferenceConfig.download_dir
        mkdir_ifnotexists(download_dir)
        yt = YouTube(url)
        video = yt.streams
        fn = strftime("video_%a_%d_%b_%Y_%H_%M_%S", gmtime())
        vid_fn = video.get_highest_resolution().download(filename=fn,output_path=download_dir)
        self.vid_file =  os.path.abspath(os.path.join(download_dir,vid_fn))
        self.progressLogger.log("URL Downloaded in "+self.vid_file)


    def extract_frames(self):
        self.progressLogger.log("Extracting frames..." )
        output_dir = os.path.abspath( self.inferenceConfig.input_dir)
        mkdir_ifnotexists(output_dir)
        ffmpeg_bin_path = self.inferenceConfig.ffmpeg_bin_path
        mkdir_ifnotexists(output_dir)
        cmd =  ffmpeg_bin_path + " -hide_banner -loglevel error -i {} -start_number 0 -vsync 0 {}/frame_%06d.png".format(
            self.vid_file, output_dir)
        #subprocess.run(cmd.split(' '))
        self.progressLogger.log('Extract cmd: '+cmd)
        run_command(cmd.split(' '))
        #os.system(cmd)
        self.progressLogger.log("Frames extracted.")




    def create_volumes(self):
        self.progressLogger.log("Creating volumes")
        mkdir_ifnotexists(self.inferenceConfig.volume_dir,self.inferenceConfig.clean_volume_dir)
        create_base_volumes(self.inferenceConfig.input_dir,self.inferenceConfig.volume_dir,
                            self.inferenceConfig.num_channels,self.inferenceConfig.file_pattern)
        self.progressLogger.log("Volumes created.")

    def run(self):
        self.download()
        self.extract_frames()
        self.create_volumes()


def download_model_export_from_gdd(model_name,model_dir):
    gdd_url = URLs[model_name]
    export_pkl = os.path.join(model_dir, str(model_name) + '_export.pkl')
    if not os.path.exists(os.path.join(model_dir,export_pkl)):
        gdd.download_file_from_google_drive(gdd_url, export_pkl)
    return export_pkl


COLORs = {0:(0,255,0),1:(255,0,0)}

def generate_result_image(preds,bar_width,bar_height=16):
    im = Image.new('RGB', (bar_width, bar_height))
    draw = ImageDraw.Draw(im)
    l = len(preds)
    prev_x = 0
    for i in range(0,l):
        x = min(int(round((i+1)*bar_width/l,0)),bar_width)
        c = COLORs[preds[i]]
        draw.rectangle((prev_x,0,x,bar_height),fill=c,outline=c)
        prev_x = x
        #if i % 50 == 0:
        #    im.show()
    return im


class R3DClassifier:

    def __init__(self,inferenceCondfig:InferenceConfig,progressLogger=None):#,st_img=None):
        self.inferenceConfig = inferenceCondfig
        self.norm_ = Normalize4d()
        #self.st_img = st_img
        self.progressLogger = progressLogger

    def load_model(self):
        if self.inferenceConfig.download_model_from_google_drive:
            model_dir = self.inferenceConfig.model_dir
            self.progressLogger.log('Loading model...')
            export_pkl = download_model_export_from_gdd(self.inferenceConfig.model_name,model_dir)
            self.learner = load_learner(model_dir,export_pkl)
            self.progressLogger.log('Model loaded.')
            self.model = self.learner.model
        else:
            if self.inferenceConfig.model_name == Models.MC3_18:
                self.model = load_mc3_model(self.inferenceConfig.model_fn)

    def infer(self):
        preds = []
        for fn in os.listdir(self.inferenceConfig.volume_dir):
            file = os.path.join(self.inferenceConfig.volume_dir,fn)
            data = torch.load(file)
            self.norm_(data)
            data.unsqueeze_(0)
            self.model.eval()
            pred = self.model(data)
            #print(pred)
            res = torch.argmax(pred)
            preds.append(res.item())
            #print('argmax',res)
            cl = self.inferenceConfig.i2c[res.item()]
            #print('result for:',file,' is ', cl )
        return preds
        
    def run(self):
        self.progressLogger.log("Loading model...")
        self.load_model()
        self.progressLogger.log("Model Loaded")
        self.progressLogger.log('Running inference...')
        preds = self.infer()
        self.progressLogger.log('Inference done!')
        return preds





def mkdir_ifnotexists(dir, clean=False):
    if clean and os.path.exists(dir): shutil.rmtree(dir)
    if os.path.exists(dir):
        return
    os.makedirs(dir)



#model = load_mc3_model('mc3_best_model_acc.pth')
#print(model)
#print(torch.__version__)
@st.cache(hash_funcs={ProgressLogger: lambda _: None})
def get_preds(cfg,progressLogger):
    cfg.model_name = Models.R3D_18
    yt = YoutubeVolumeCreator(cfg, progressLogger)
    yt.run()
    classifier = R3DClassifier(inferenceCondfig=cfg, progressLogger=progressLogger)  # , st_img=st_img)
    preds = classifier.run()
    return preds


def main():
    st.subheader("Enter the URL:")
    url = st.text_input(label='URL')
    session_state = get(status = 0, url='')
    # 'https://www.youtube.com/watch?v=oRQyu66zGE4'
    preds=[]
    cfg = InferenceConfig()
    session_state.status
    type(session_state.status)
    if url != '':
        if session_state.status == 0:
            cfg.youtube_url = url
            session_state.url = url
            progress_log_text = st.empty()
            progressLogger = ProgressLogger(progress_log_text)
            preds = get_preds(cfg,progressLogger)
            session_state.status = 1
        else:    
            frame_width = get_frame_width(cfg.input_dir)
            img = generate_result_image(preds, frame_width)
            event = st_player(cfg.youtube_url, events=['onProgress'], progress_interval=200)
            st.image(img)
            st.slider('', 0.0, 1.0, float(event.data['played']), 0.01)
            session_state.status

main()
#os.makedirs('/tmp/frames')
#cfg = InferenceConfig()
#mkdir_ifnotexists(cfg.input_dir)
#st.write(os.path.exists('/tmp/frames'))
