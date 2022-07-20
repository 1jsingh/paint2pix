import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
import glob

import cv2
from cv2 import stylization
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
#from svgpathtools import parse_path

from predict import load_model, predict_image_completion, decode_latent, load_faceid_model, identity_constrained_latent_pred, encoder_based_id_edit

from utils.common import tensor2im
from PIL import ImageColor
import torch
import cv2
import paddlehub as hub
import glob as glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def display_alongside_batch(img_list, resize_dims=(256,256)):
    res = np.concatenate([np.array(img.resize(resize_dims)) for img in img_list], axis=1)
    return Image.fromarray(res)


def main():
    if 'button_id' not in st.session_state:
        st.session_state['button_id'] = ''
    if 'color_to_label' not in st.session_state:
        st.session_state['color_to_label'] = {}
    PAGES = {
        "Real image editing": paint2pix_demo,
        "Progressive image synthesis": paint2pix_demo,
        "Artistic content generation": paint2pix_demo,
    }
    st.sidebar.subheader("Paint2Pix Demos")
    page = st.sidebar.selectbox("Demo:", options=list(PAGES.keys()))
    PAGES[page](page.lower())

    # with st.sidebar:
    #     st.markdown("---")
    #     st.markdown(
    #         '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
    #         unsafe_allow_html=True,
    #     )
    #     st.markdown(
    #         '<div style="margin: 0.75em 0;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
    #         unsafe_allow_html=True,
    #     )


def paint2pix_demo(demo_type='real image editing'):
    st.sidebar.header("Configuration")
    
    # Specify canvas parameters in application
    experiment_type = 'ffhq'#st.sidebar.selectbox('Experiment Type', ('ffhq','cars_encode'))
    
    if experiment_type == 'ffhq':
        resize_dims = (256,256)
    elif experiment_type == 'cars_encode':
        resize_dims = (192,256)
    
    # st.sidebar.subheader('Intelli-paint Config:')
    # canvas_frame_id = st.sidebar.slider("Painting Canvas Frame id: ", 1, 100, 100)
    # num_paint_gd_iter = st.sidebar.slider("Painting GD iterations: ", 0, 100, 0)
    
    st.sidebar.markdown('---')
    st.sidebar.header('User brush stroke parameters:')
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 40)
    stroke_opacity = st.sidebar.slider("Stroke opacity: ", 0, 100, 80)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#E8BEAC")
    
    
    r,g,b = ImageColor.getcolor(stroke_color, "RGB")
    stroke_color_with_opacity = "rgba({},{},{},{})".format(r,g,b,stroke_opacity/100.0)
    
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    st.sidebar.subheader('Restyle Config:')
    restyle_iter = st.sidebar.slider("restyle iter: ", 1, 10, 5)
    restyle_select_iter = st.sidebar.slider("restyle select iter: ", 1, restyle_iter, restyle_iter)
    
    # st.sidebar.subheader('Multi-modal Prediction Params:')
    multi_modal = False #st.sidebar.checkbox('Multi-modal')
    num_multi_output = 5#st.sidebar.slider("Number of multi-modal outputs: ", 1, 10, 5)
    
    # --------------------------------------------------------------------------------------------#
    #  main app body
    #  introduction
    st.markdown(
    """
    Paint2Pix: Interactive Painting based Image Synthesis and Editing
    """
    )
    st.markdown("""<img src="https://1jsingh.github.io/assets/publications/images/paint2pix.png" alt="Streamlit logo" height="280"><br><br>""", unsafe_allow_html=True)

    # --------------------------------------------------------------------------------------------#
    # canvas and image prediction
    if demo_type=='real image editing':
        st.header('Real Image Editing')
        st.markdown("Start editing by using a real image input or create your own image using brushstroke inputs")
    elif demo_type=='progressive image synthesis':
        st.header('Progressive Image Synthesis')
        st.markdown("Express your inner ideas ... synthesize your desired output image using just coarse scribbles")
    elif demo_type=='artistic content generation':
        st.header('Artistic Content Generation')
        st.markdown("Unleash your inner artist ... create high artistic paintings using just coarse scribbles")

    # choose input image from a list of custom image inputs for real image editing
    if demo_type=='real image editing':
        input_image_list = sorted(list(glob.glob('input-images/*')))
        input_image_options = [Image.open(x).resize((128,128)) for x in input_image_list]
        st.image(input_image_options, caption = ["Input image {}".format(i+1) for i in range(len(input_image_list))])
        selected_image = st.selectbox("Select image input:",["Input image {}".format(i+1) for i in range(len(input_image_list))]+["Custom Input"])
    else:
        selected_image = "Paint from scratch"
    
    bg_image_container = st.empty()
    if selected_image == "Custom Input":
        bg_image = bg_image_container.file_uploader("Input image:", type=["png", "jpg"])
        # bg_image = bg_file_upload.getvalue().decode("utf-8") if bg_file_upload is not None else None
    elif selected_image.startswith('Input image'):
        bg_image = input_image_list[int(selected_image[-1])-1]
    else:
        bg_image = None

    get_value = lambda x: x if x is None or isinstance(x,str) else x.getvalue()# if 'name' in x else x
    # using canvas prediction as initialization of canvas
    if 'bg_img' not in st.session_state or get_value(st.session_state.bg_image) != get_value(bg_image):
        use_canvas_pred = False #st.checkbox('Use Canvas Prediction')
        if 'bg_img' in st.session_state and get_value(st.session_state.bg_image) != get_value(bg_image):
            #st.text("{}...{}".format(st.session_state.bg_image,bg_image))
            #st.text(st.session_state.bg_image.getvalue()==bg_image.getvalue())
            # st.text("reset demo state ... ")
            # Reset Session state
            for key in st.session_state.keys():
                # st.text(key)
                del st.session_state[key]
            time.sleep(1)
        st.session_state.bg_image = bg_image
        if bg_image is None and 'canvas_pred' in st.session_state and use_canvas_pred:
            canvas_pred = cv2.resize(st.session_state['canvas_pred'][canvas_frame_id-1],resize_dims[::-1])
            bg_img = Image.fromarray(canvas_pred)
        elif bg_image is not None:
            bg_img = Image.open(bg_image).convert('RGB')
            st.session_state['prediction'] = Image.open(bg_image).convert('RGB')
        else:
            bg_img = None
            
        if bg_img is None:
            bg_img_ = np.zeros((resize_dims[0],resize_dims[1],3)).astype('float32')
            bg_img = Image.new('RGB',(256,256))
        else:
            bg_img_ = np.float32(bg_img)/255.#.convert('RGBA')
            bg_img_ = cv2.resize(bg_img_,resize_dims[::-1])
        st.session_state.bg_img = bg_img
        st.session_state.bg_img_ = bg_img_
        st.session_state.real_image_input = bg_img
        

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Canvas')
        stylized_output = True if demo_type=='artistic content generation' else False
        use_image_pred_as_input = st.button('Use Image Prediction as Canvas')
        if use_image_pred_as_input and 'prediction' in st.session_state:
            bg_img = st.session_state.id_constrained_pred
            bg_img_ = np.float32(bg_img)/255.#.convert('RGBA')
            bg_img_ = cv2.resize(bg_img_,resize_dims[::-1])
            st.session_state.bg_img = bg_img
            st.session_state.bg_img_ = bg_img_
            st.session_state.input_latent = st.session_state.output_latent
            st.session_state.real_image_input = bg_img
            # compute w1'
            # net, transform, opts = load_model(experiment_type)
            # result_images, multi_out, latents = predict_image_completion(st.session_state.real_image_input, net, transform, opts, multi_modal=False, experiment_type=experiment_type, resize_dims=resize_dims,num_multi_output=num_multi_output, n_iters=restyle_iter,latent_mask=None,mix_alpha=None, id_constrain=False, target_id_feat=None)
            # st.session_state.input_latent_ = latents[0][restyle_select_iter-1]
            st.session_state.input_latent_ = st.session_state.input_latent


        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color_with_opacity,
            background_color=bg_color,
            background_image=st.session_state.bg_img,
            update_streamlit=realtime_update,
            height=resize_dims[0],
            width=resize_dims[1],
            drawing_mode=drawing_mode,
            display_toolbar=st.sidebar.checkbox("Display drawing toolbar", True),
            key="full_app",
        )
            
            
    with col2:
        # image completion predictions
        st.subheader('Image Prediction')
        id_constrain = False
        if True:#canvas_result.image_data is not None:
            # Initialization
            if 'prediction' not in st.session_state:
                st.session_state['prediction'] = Image.fromarray(np.zeros((resize_dims[0],resize_dims[1],3)).astype('uint8'))
            if 'multi_modal_prediction' not in st.session_state:
                st.session_state['multi_modal_prediction'] = Image.fromarray(np.zeros((resize_dims[0],resize_dims[1]*7,3)).astype('uint8'))
            if 'canvas_pred' not in st.session_state:
                st.session_state['canvas_pred'] = np.zeros((100,resize_dims[0],resize_dims[1],3)).astype('uint8')
            if 'restyle_prediction' not in st.session_state:
                st.session_state['restyle_prediction'] = np.zeros((10,resize_dims[0],resize_dims[1],3)).astype('uint8')
            if 'canvas' not in st.session_state:
                st.session_state['canvas'] = Image.fromarray(np.zeros((resize_dims[0],resize_dims[1],3)).astype('uint8'))
                
            img_pred_button = st.button('Predict')
            if img_pred_button:# or 'id_constrained_pred' not in st.session_state:
                with st.spinner('Computing image completion prediction ...'):
                    net, transform, opts = load_model(experiment_type,id_constrain,stylized_output=stylized_output)
                    if canvas_result.image_data is None:
                        input_image = bg_img
                    else:
                        painting = np.float32(canvas_result.image_data)/255.
                        alpha = cv2.cvtColor(painting[:,:,-1],cv2.COLOR_GRAY2RGB)
                        painting_fg = painting[:,:,:3]
                        foreground = cv2.multiply(alpha, painting_fg)
                        background = cv2.multiply(1.0 - alpha, st.session_state.bg_img_)
                        outImage = cv2.add(foreground, background)
                        input_image = Image.fromarray(np.uint8(255*outImage))
                    st.session_state['canvas'] = input_image
                    if multi_modal:
                        latent_mask = list(range(mixing_layers[0],mixing_layers[1]+1))
                        mix_alpha = mix_alpha/100
                    else:
                        latent_mask, mix_alpha  = None, None

                    result_images, multi_out, latents = predict_image_completion(input_image, net, transform, opts, multi_modal=multi_modal, experiment_type=experiment_type, resize_dims=resize_dims,num_multi_output=num_multi_output, n_iters=restyle_iter,latent_mask=latent_mask,mix_alpha=mix_alpha, id_constrain=id_constrain, target_id_feat=None)
                st.session_state.restyle_prediction = result_images
                st.session_state.latents = latents[0]
                #st.session_state.prediction = result_images[res]
                if multi_modal:
                    st.session_state['multi_modal_prediction'] = multi_out#display_alongside_batch(multi_out[0:],resize_dims=(256,256))
            if multi_modal:
                bbox = (select_prediction*resize_dims[1],0,(select_prediction+1)*resize_dims[1],resize_dims[0])
                st.session_state.prediction = st.session_state['multi_modal_prediction'].crop(bbox)
            else:
                st.session_state.prediction = st.session_state.restyle_prediction[restyle_select_iter-1]
            output_image_placeholder = st.empty()
    
    with st.expander('Control Edit Strength'):
        st.markdown("The user can use extrapolation of edit strength in order to achieve desired output attributes. This is helpful in achieving semantic image edits *e.g.,* aging which are otherwise difficult to describe using just coarse user scribbles.")
        col1, col2 = st.columns(2)
        with col1:
            threshold_beta = st.number_input('threshold-beta',value=1.0,step=0.5)
        with col2:
            edit_alpha = st.number_input('edit-alpha',value=1.0,step=0.2)
        
        num_style_layers = 18 if experiment_type=='ffhq' else 16
        editable_layers_global = st.slider('Editable StyleGAN layers:',0, num_style_layers-1, (0,num_style_layers-1))
        
        if img_pred_button:# or 'edited_img' not in st.session_state:
            st.session_state.output_latent = st.session_state.latents[restyle_select_iter-1]
            if 'input_latent' not in st.session_state:
                st.session_state.input_latent = st.session_state.input_latent_ = st.session_state.output_latent
            
            mask = np.arange(editable_layers_global[0],editable_layers_global[1]+1)
            latent_mask = np.zeros(num_style_layers)
            latent_mask[mask] = 1.0
            
            # net, transform, opts = load_model(experiment_type,stylized_output=stylized_output)
            delta_w = st.session_state.output_latent - st.session_state.input_latent_
                    
            delta_w = np.expand_dims(latent_mask,-1) * np.clip(delta_w,-threshold_beta,threshold_beta) 
            w0 = st.session_state.input_latent#st.session_state.latents[restyle_select_iter-1]
            w1 = w0 + delta_w * edit_alpha
            edited_img = decode_latent(w1, net, opts, experiment_type='ffhq', resize_dims=(256,256), truncation=1.)
            st.session_state.edited_img = edited_img
            st.session_state.output_latent = w1
            st.image([st.session_state.prediction.resize((200,200)),st.session_state.edited_img.resize((200,200))],["before","after"])

    with st.expander('Identity Correction Config'):
        num_id_gd_iter = st.slider('Number of iterations:',0,1000,20)
        col1, col2 = st.columns(2)
        with col1:
            lambda_reg = st.number_input('lambda id-reg',value=0.01)
        with col2:
            editable_layers = st.slider('StyleGAN Editable Layers:',0, 17, (0,8))
            mask = np.arange(editable_layers[0],editable_layers[1]+1)
            latent_mask = np.zeros(18)
            latent_mask[mask] = 1.0
            
        col1, col2 = st.columns(2)
        with col1:
            use_encoder = st.checkbox('Use Identity Encoder',True)
        with col2:
            num_id_enc_iter = st.slider('Number of identity-encoding steps:',1,5,1)

        if img_pred_button:# or 'id_constrained_pred' not in st.session_state:
            id_loss_func = load_faceid_model()
            # net, transform, opts = load_model(experiment_type,stylized_output=stylized_output)
            x =  st.session_state.output_latent 
            target_img = st.session_state.real_image_input
            if use_encoder:
                original_img = st.session_state.edited_img
                id_net,_,_ = load_model(experiment_type,id_constrain=True,stylized_output=stylized_output)
                id_constrained_pred, delta_w = encoder_based_id_edit(original_img, x, target_img, id_net, transform, opts, latent_mask=latent_mask, num_id_iter=num_id_enc_iter)
            else:
                id_constrained_pred, loss_log, delta_w = identity_constrained_latent_pred(x, target_img, net, transform, opts, id_loss_func, input_code=True, n_iter=num_id_gd_iter, lr=5e-3, lambda_id=1.0, lambda_reg=lambda_reg,latent_mask=latent_mask, lambda_l2=1e1)
                st.session_state.id_constrained_loss_log = loss_log
            st.session_state.id_constrained_pred = id_constrained_pred
            st.session_state.output_latent = st.session_state.output_latent + delta_w[0]
        if 'id_constrained_pred' in st.session_state:
            output_image_placeholder.image(st.session_state.id_constrained_pred)
            st.image([st.session_state.real_image_input.resize((200,200)),st.session_state.edited_img.resize((200,200)),st.session_state.id_constrained_pred.resize((200,200))],["Identity Image","w/o Identity Encoder","with Identity Encoder"])


def png_export():
    st.markdown(
        """
    Realtime update is disabled for this demo. 
    Press the 'Download' button at the bottom of canvas to update exported image.
    """
    )
    try:
        Path("tmp/").mkdir()
    except FileExistsError:
        pass

    # Regular deletion of tmp files
    # Hopefully callback makes this better
    now = time.time()
    N_HOURS_BEFORE_DELETION = 1
    for f in Path("tmp/").glob("*.png"):
        st.write(f, os.stat(f).st_mtime, now)
        if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
            Path.unlink(f)

    if st.session_state["button_id"] == "":
        st.session_state["button_id"] = re.sub("\d+", "", str(uuid.uuid4()).replace("-", ""))

    button_id = st.session_state["button_id"]
    file_path = f"tmp/{button_id}.png"

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data.encode()).decode()
        except AttributeError:
            b64 = base64.b64encode(img_data).decode()

        dl_link = (
            custom_css
            + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
        )
        st.markdown(dl_link, unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Paint2Pix Demo", page_icon=":pencil2:"
    )
    st.title("Paint2Pix: Demo")
    main()
