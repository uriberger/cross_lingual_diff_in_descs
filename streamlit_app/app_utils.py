import streamlit as st
from st_lazy_clickable_images import clickable_images
import base64
import aiohttp
import asyncio

download_images_button_ind = 0


async def fetch_image(session, url):
    """Asynchronously fetch an image and encode it in base64."""
    async with session.get(url) as response:
        content = await response.read()
        return base64.b64encode(content).decode()


async def download_images(image_file_urls):
    """Asynchronously download multiple images."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, url) for url in image_file_urls]
        return await asyncio.gather(*tasks)


def plot_clickable_images(iids):
    global download_images_button_ind

    image_names = [f"{hex(iid)[2:].zfill(16)}.jpg" for iid in iids]
    st.download_button(
        "List of filenames of the images",
        "\n".join(image_names),
        key=f"download_images{download_images_button_ind}",
    )
    download_images_button_ind += 1
    image_file_urls = [
        f"https://storage.googleapis.com/xm3600/{hex(iid)[2:].zfill(16)}.jpg"
        for iid in iids
    ]

    # Use asyncio to run the asynchronous download_images function
    images = asyncio.run(download_images(image_file_urls))
    images = [f"data:image/jpeg;base64,{encoded}" for encoded in images]

    # Pass the image data as a list of base64-encoded strings
    clicked = clickable_images(
        images,
        titles=[f"Image #{str(i)}" for i in range(len(image_file_urls))],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )

    return clicked
