from aiogram import types
from aiogram.utils.exceptions import MessageNotModified
from asyncio import get_running_loop

from misc.labels import labels
from misc.pipe import answer_image


user_selections = {}

async def start_command(message: types.Message) -> None:      
    reply_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    reply_keyboard.add(labels['guide'])

    await message.answer(labels['greeting'].format(message.from_user.first_name), reply_markup=reply_keyboard)

async def print_guide(message: types.Message) -> None:
    await message.answer(labels['guide_text'])

async def get_settings_inline(models, tasks, model_default, tasks_default) -> types.InlineKeyboardMarkup:
    inline_markup = types.InlineKeyboardMarkup()

    inline_markup.add(*(
        types.InlineKeyboardButton(model_name + (' ✅' if model_default == model_name else ''), callback_data=f'settings_model_text_id_select: {model_name}')
        for model_name in models
    ))
    inline_markup.add(*(
        types.InlineKeyboardButton(task_name + (' ✅' if tasks_default == task_name else ''), callback_data=f'settings_task_select: {task_name}')
        for task_name in tasks
    ))
    inline_markup.add(types.InlineKeyboardButton(labels['generate'], callback_data=f'start_generating'))

    return inline_markup

async def print_settings(message: types.Message) -> None:
    models = labels['models']
    tasks = labels['tasks']

    user_id = message.from_user.id
    if user_id not in user_selections:
        user_selections[user_id] = {'model': models[0], 'task': tasks[0]}
    
    model_default = user_selections[user_id]['model']
    tasks_default = user_selections[user_id]['task']

    inline_markup = await get_settings_inline(models, tasks, model_default, tasks_default)

    last_photo_file_id = message.photo[-1].file_id
    await message.answer_photo(photo=last_photo_file_id, caption=labels['settings_text'], reply_markup=inline_markup)

async def change_settings(callback: types.CallbackQuery) -> None:
    obj, target = callback.data.split(' ')

    models, tasks = labels['models'], labels['tasks']
    
    user_id = callback.from_user.id
    if user_id not in user_selections:
        user_selections[user_id] = {'model': models[0], 'task': tasks[0]}
    
    model_default = user_selections[user_id]['model']
    tasks_default = user_selections[user_id]['task']

    await callback.answer()
    if target == model_default or target == tasks_default:
        return

    if obj == 'settings_model_text_id_select:':
        model_default = target
    elif obj == 'settings_task_select:':
        tasks_default = target
    
    user_selections[user_id] = {'model': model_default, 'task': tasks_default}

    inline_markup = await get_settings_inline(models, tasks, model_default, tasks_default)

    try:
        await callback.message.edit_reply_markup(inline_markup)
    except MessageNotModified:
        pass

async def send_large_message(message: types.Message, text: str, max_length: int = 4096) -> None:
    for start in range(0, len(text), max_length):
        end = start + max_length
        await message.answer(text[start:end], reply=True, allow_sending_without_reply=True)

async def generate_output(callback: types.CallbackQuery) -> None:
    user_id = callback.from_user.id
    models, tasks = labels['models'], labels['tasks']
    if user_id not in user_selections:
        user_selections[user_id] = {'model': models[0], 'task': tasks[0]}

    file_id = callback.message.photo[-1].file_id
    
    file = await callback.bot.get_file(file_id)
    file_path = file.file_path

    base_url = 'https://api.telegram.org/file/bot'
    url = f"{base_url}{callback.bot._token}/{file_path}"

    await callback.answer(labels['start_generating'])

    model, task = user_selections[user_id]['model'], labels['tasks'].index(user_selections[user_id]['task'])

    loop = get_running_loop()
    output = await loop.run_in_executor(None, answer_image, url, model, task)

    await send_large_message(callback.message, f"{labels['generation_result']}\n\n" + output)

async def message_close(callback: types.CallbackQuery) -> None:
    await callback.message.edit_text(labels['cancel'])

# async def other_message_handler(message: types.Message) -> None:  
#     await message.answer(labels['unknown_command'])

def register_handlers(dp) -> None:
    dp.register_message_handler(start_command, commands=['start', 'reload'])
    dp.register_message_handler(print_guide, lambda m: m.text == labels['guide'])
    dp.register_message_handler(print_settings, content_types=['photo'])
    # dp.register_message_handler(other_message_handler)

    dp.register_callback_query_handler(message_close, lambda m: m.data == 'message_close')
    dp.register_callback_query_handler(change_settings, lambda m: m.data.startswith('settings_model_text_id_select:') or m.data.startswith('settings_task_select:'))
    dp.register_callback_query_handler(generate_output, lambda m: m.data == 'start_generating')