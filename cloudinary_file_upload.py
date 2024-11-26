import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import os

# Configuration     
cloudinary.config( 
    cloud_name = "ducl7cu1b", 
    api_key = "447252138741476", 
    api_secret = "3EHV8KV91eQ5dpU1SbcCpInTmpc", # Ensure this is secure in production
    secure=True
)


def upload_img(file_):
    

    upload_result = cloudinary.uploader.upload(
        'uploads/'+file_,
        folder="post_det"   # Specify the folder in Cloudinary
    )

        # Print the URL of the uploaded image
    file_url=upload_result["secure_url"]

    os.remove('uploads/'+file_)

    return file_url


if __name__=="__main__":
    cloud_upload_img()