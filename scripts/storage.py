import base64
from io import BytesIO
import os
import re
import sys
from modules import shared, script_callbacks
import modules.scripts as scripts
from modules.shared import opts
import gradio as gr
from pymongo import MongoClient

if hasattr(opts, "db_storage_database_host"):
    mongo_host = opts.db_storage_database_host
else:
    mongo_host = os.environ.get('DB_HOST', 'localhost')
if hasattr(opts, "db_storage_database_port"):
    mongo_port = opts.db_storage_database_port
else:
    mongo_port = int(os.environ.get('DB_PORT', 27017))
if hasattr(opts, "db_storage_database_user"):
    mongo_username = opts.db_storage_database_user
else:
    mongo_username = os.environ.get('DB_USER', '')
if hasattr(opts, "db_storage_database_password"):
    mongo_password = opts.db_storage_database_password
else:
    mongo_password = os.environ.get('DB_PASS', '')

creds = f"{mongo_username}:{mongo_password}@" if mongo_username and mongo_password else ""
client = MongoClient(f"mongodb://{creds}{mongo_host}:{mongo_port}/")
savedFiles = []

def get_collection(database_name, collection_name):
    db = client[database_name]
    collection = db[collection_name]
    return collection

class Scripts(scripts.Script):
    def title(self):
        return "Mongo Storage"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        checkbox_save_to_db = gr.inputs.Checkbox(label=f"Save to DB", default=True)
        return [checkbox_save_to_db,]
    
    def postprocess(self, p, processed, checkbox_save_to_db):
        global savedFiles

        collection = get_collection(opts.db_storage_default_database, opts.db_storage_default_collection) if checkbox_save_to_db else None
        if collection is None:
            return True
        
        # print(dir(p))
        # print(dir(processed))
        # print(savedFiles)
        hasError = False
        info = re.findall(r"Steps:.*$", processed.info, re.M)[0]
        usingControlNet = False
        controlNetLocation = str(info).find("ControlNet")
        input_dict = {}
        if(controlNetLocation > 0): # check if has ControlNet parameters since they include commas
            info = str(info)[0:controlNetLocation-2]
            usingControlNet = True
        if opts.db_storage_debug_mode:
            print("Info: ", str(info))
        try:
            for item in str(info).split(", "):
                if(item.find(':') > -1):
                    inpItems = item.split(':')
                    input_dict[inpItems[0]] = inpItems[1]
        except ValueError:
            print("==> SD DB Storage: Error parsing Extra Details. Skipping.")
            hasError = True
        
        if not hasError:
            numImages = len(processed.images)
            images = processed.images.copy()
            curFilenameIndex = 0

            savedFilenames = []
            hasPreprocessed = False
            for fn in savedFiles: # clear out pre-processed images
                    if(fn.find('restoration') < 0 and fn.find('highres') < 0):
                        savedFilenames.append(fn)
                    else:
                        hasPreprocessed = True

            # TODO: may not handle batch controlnets with hires fix, need testing or better way to filter out previews and grids
            if usingControlNet:
                removedImage = images.pop(-1) # if using ControlNet, skip last image/model preview
                if(hasPreprocessed):
                    removedImage = images.pop(-1) # if using ControlNet and HiRes, again skip last image/model preview since it makes 2
                numImages = len(images)
            if (numImages > 1):
                removedImage = images.pop(0) # if grid/batch, remove first image
                numImages = len(images)
            image_mode = p.__class__.__name__.replace('StableDiffusionProcessing','')

            try:    
                for index in range(0, numImages):
                    if(index < numImages):
                        image = images[index]
                        sizeMultiplier = 1
                        if("Hires upscale" in input_dict):
                            sizeMultiplier = int(input_dict["Hires upscale"])
                        insertedImage = {
                            "mode": image_mode,
                            "prompt": processed.all_prompts[index], 
                            "negative_prompt": processed.negative_prompt, 
                            "steps": int(input_dict["Steps"]), 
                            "seed": int(processed.all_seeds[index]), 
                            "sampler": input_dict["Sampler"],
                            "cfg_scale": float(input_dict["CFG scale"]), 
                            "model": input_dict["Model"],
                            "model_hash": input_dict["Model hash"],
                            "size": tuple(map(lambda x: int(x)*sizeMultiplier, input_dict["Size"].split("x")))
                        }

                        if(len(savedFilenames) > 0):
                            insertedImage["filename"] = os.path.basename(savedFilenames[curFilenameIndex])
                            insertedImage["filepath"] = os.path.dirname(savedFilenames[curFilenameIndex])

                        ## TODO Better parse of the ControlNet options
                        if(usingControlNet):
                            insertedImage["ControlNet"] = True

                        # Quick and dirty in case of something like sd-dynamic-prompts, though only catches default characters
                        if("__" in processed.prompt):
                            insertedImage["initial_prompt"] = processed.prompt

                        if opts.db_storage_save_full_image:
                            buffer = BytesIO()
                            image.save(buffer, "png", quality='keep')
                            image_file_size = buffer.tell()
                            image_bytes = buffer.getvalue()
                            insertedImage["image"] = image_bytes
                            insertedImage["filesize"] = image_file_size

                        collection.insert_one(insertedImage)
                        curFilenameIndex = curFilenameIndex + 1
            except Exception as e:
                print("SD DB Storage ==> Error parsing data for database. Received data outside of scope. Skipped.")
                if opts.db_storage_debug_mode:
                    print("Exception:", e)
            savedFiles = []
            insertedImage = {}
            return True

def on_before_image_saved(params : script_callbacks.ImageSaveParams):
    global savedFiles
    
    if(params.filename.find('grid') < 0):
        savedFiles.append(params.filename)

def on_ui_settings():
    # [current setting_name], [default], [label], [component], [component_args]
    storage_options = [
        ("db_storage_database_host", "localhost", "Database Host"),
        ("db_storage_database_port", "27017", "Database Port"),
        ("db_storage_database_user", "", "Database Username"),
        ("db_storage_database_password", "", "Database Password"),
        ("db_storage_default_database", "StableDiffusion", "Default Database Name"),
        ("db_storage_default_collection", "Images", "Default Collection Name"),
        ("db_storage_save_full_image", True, "Save Full Image Data to DB", gr.Checkbox, {"interactive": True}),
        ("db_storage_debug_mode", False, "Debug Mode", gr.Checkbox, {"interactive": True}),
    ]

    section = ('db-storage', "DB Storage")
    for cur_setting_name, *option_info in storage_options:
        shared.opts.add_option(cur_setting_name, shared.OptionInfo(*option_info, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_image_saved(on_before_image_saved)
