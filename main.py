import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command

# Токен бота
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Создание объектов бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Логирование
logging.basicConfig(level=logging.INFO)

# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привет! Я ваш Telegram-бот.")

# Обработчик текстовых сообщений
@dp.message()
async def echo(message: Message):
    await message.answer(f"Вы сказали: {message.text}")

# Главная функция запуска бота
async def main():
    await dp.start_polling(bot)

# Запуск бота
if __name__ == "__main__":
    asyncio.run(main())