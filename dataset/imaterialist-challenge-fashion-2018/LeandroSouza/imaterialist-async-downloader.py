import aiofiles
import aiohttp
import asyncio
import async_timeout
import os
import numpy as np
import json


async def download(session, url, idx, output_folder):
    """Performs async download from a image and saves it.
    Args:
        session (ClientSession): aiohttp session.
        url (string): url of image.
        idx (int): index of image.
        output_folder (string): path to save image.
    """
    async with session.get(url) as response:
        async with aiofiles.open(output_folder + str(idx) + '.jpeg', 'wb') as fd:
            while True:
                chunk = await response.content.read(1024 * 1024)
                if not chunk:
                    break
                await fd.write(chunk)
        print("Image:{}, Operation:{}".format(idx, output_folder.split('/')[-2]))
        return await response.release()


async def bound_download(sem, session, url, idx, output_folder):
    """Limit of the number async requests using Semaphores.
    Args:
        session  (ClientSession): aiohttp session.
        url (string): url of image.
        idx (int): index of image.
        output_folder (string): path to save image.
    """
    async with sem:
        await download(session, url, idx, output_folder)


async def load_data(output_folder, json_path):
    """Load the documents from the json file and start the download process.
    Args:
        output_folder (string): images output folder.
        json_path (string): path to json file (train, validation and test).
    """

    # create instance of Semaphore
    sem = asyncio.Semaphore(2000)
    tasks = []

    with open(json_path, 'r') as f:
        documents = json.load(f)

    images = documents.get('images')

    async with aiohttp.ClientSession(loop=loop) as session:
        for idx, img in enumerate(images):
            task = asyncio.ensure_future(bound_download(
                sem, session, img.get('url'), idx, output_folder))
            tasks.append(task)

        responses = asyncio.gather(*tasks)
        await responses


async def main(loop):

    dataset = os.environ.get('DATASET')
    sub_folder = 'dataset/'

    train_output_folder = dataset + sub_folder + 'train/'
    validation_output_folder = dataset + sub_folder + 'validation/'
    test_output_folder = dataset + sub_folder + 'test/'

    json_train = dataset + 'train.json'
    json_validation = dataset + 'validation.json'
    json_test = dataset + 'test.json'

    if not os.path.exists(train_output_folder):
        os.makedirs(train_output_folder)

    if not os.path.exists(validation_output_folder):
        os.makedirs(validation_output_folder)

    if not os.path.exists(test_output_folder):
        os.makedirs(test_output_folder)

    await load_data(train_output_folder, json_train)
    await load_data(validation_output_folder, json_validation)
    await load_data(test_output_folder, json_test)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
