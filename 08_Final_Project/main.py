from aiogram import Bot, Dispatcher
from aiogram.utils import executor
from misc.handlers import register_handlers
from misc.utils import set_up_images_tmp
set_up_images_tmp()

import logging
import os, dotenv

dotenv.load_dotenv()
logging.basicConfig(level=logging.WARNING)

bot = Bot(token=os.getenv('BOT_TOKEN'))
dp = Dispatcher(bot)
register_handlers(dp)

if __name__ == '__main__':
    while True:
        executor.start_polling(dp, skip_updates=True)