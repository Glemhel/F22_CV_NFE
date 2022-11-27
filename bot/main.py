from pathlib import Path
# for modules imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

import telebot
from config import token
from inference import *
from telebot import custom_filters, types
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

"""
Init for NFE ML model components
"""

model = NFEModel(PROJECT_ROOT)


"""
Init telegram bot
"""

bot = telebot.TeleBot(token, parse_mode="HTML")
bot.remove_webhook()

user_data = {}

"""
Telegram bot functions
"""


def generate_markup_manipulation_initial(user_id):
    markup = InlineKeyboardMarkup()
    id_to_emoji = {
         0: '',
         1: '\U00002705',
         2: '\U0001F512'
    }
    emojis = [id_to_emoji[user_data[user_id][prop]] for prop in model.properties]
    for emoji, prop in zip(emojis, model.properties):
        markup.row(InlineKeyboardButton(prop + emoji, callback_data='cb_' + prop))
    markup.row(
        *[
            InlineKeyboardButton('Go!', callback_data='cb_proceed')
        ]
    )
    return markup

def generate_markup_manipulation_option(property):
    markup = InlineKeyboardMarkup()
    for opt in model.property_to_options[property]:
        markup.add(InlineKeyboardButton(opt, callback_data='attr_' + opt))
    return markup

"""
Latent space manipulation
"""
def change_attribute(user_id, feature, value, preserved_features):
    if len(preserved_features) == 0:
        preserved_features = None
    z = user_data[user_id]['latent_vector']
    # image manipulation
    medias = model.change_image(str(user_id), z, feature, value, preserved_features)
    medias_typed = [types.InputMediaPhoto(x) for x in medias]
    medias_typed[0].caption = 'Your face changes! Try again with different parameters! /change'
    bot.send_media_group(user_id, medias_typed)

"""
Buttons various callbacks.
"""
@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    userid = call.from_user.id
    if call.data[3:] in model.properties:
        prop = call.data[3:]
        val = (user_data[userid][prop] + 1) % 3
        user_data[userid][prop] = val
        bot.edit_message_reply_markup(
            chat_id=call.from_user.id,
            message_id=call.message.message_id,
            reply_markup=generate_markup_manipulation_initial(call.from_user.id)
        )
        bot.answer_callback_query(call.id)
    elif call.data == 'cb_proceed':
        cnt = 0
        chosen_option = None
        chosen_option_locked = []
        for key in model.properties:
            if user_data[userid][key] == 1:
                cnt += 1
                chosen_option = key
            elif user_data[userid][key] == 2:
                chosen_option_locked.append(key)
        user_data[userid]['feature'] = chosen_option
        user_data[userid]['preserved'] = chosen_option_locked

        replytext = None
        if cnt > 1:
            replytext = 'Only single option should be marked for change!'
        elif cnt == 0:
            replytext = 'At least one option should be marked for change!'
        if replytext is not None:
            bot.answer_callback_query(callback_query_id=call.id, show_alert=True, text=replytext)
        else:
            bot.answer_callback_query(callback_query_id=call.id)
            s = f'Attribute for change: {chosen_option}\nAttributes to preserve: {None if len(chosen_option_locked) == 0 else chosen_option_locked}\n' \
                f'Select option to change to:'
            bot.send_message(userid, s, reply_markup=generate_markup_manipulation_option(chosen_option))
    elif call.data[:5] == 'attr_':
        feature = user_data[userid]['feature']
        value = call.data[5:]
        preserved_features = user_data[userid]['preserved']
        change_attribute(userid, feature, value, preserved_features)
        bot.answer_callback_query(call.id)


"""
Perform GAN inversion on photo sent.
"""
@bot.message_handler(content_types=['photo'])
def upload_and_inversion(message):
    if not torch.cuda.is_available():
        bot.send_message(message.chat.id, 'Cuda is not available. Please generate random face with /face instead!')
        return
    # get photo from user
    bot.send_message(message.chat.id, 'Doing GAN inversion on your image. Please wait...')
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    user_image_path = f'{message.chat.id}-user-image.png'
    gan_image_path = f'{message.chat.id}-gan-image.png'
    with open(user_image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    # apply gan inversion - very expensive, only on gpu!!
    z = model.gan_inversion(user_image_path)
    # fill user informatoin
    user_data[message.chat.id] = {}
    user_data[message.chat.id]['latent_vector'] = z.cpu().detach().numpy()
    model.generate_and_save_image(z, gan_image_path)
    user_data[message.chat.id]['gan_image_path'] = gan_image_path
    user_data[message.chat.id]['original_image_path'] = user_image_path
    # generate reply for user
    pics = model.picture_array(
        [user_data[message.chat.id]['original_image_path'], user_data[message.chat.id]['gan_image_path']]
    )
    medias_typed = [types.InputMediaPhoto(x) for x in pics]
    medias_typed[0].caption = 'Initial face and Face after GAN inversion!'
    # send before/after gan inversion images
    bot.send_media_group(message.chat.id, medias_typed)
    bot.send_message(message.chat.id, 'Now send /change to display possible changes!')

"""
Image manipulation entry point
"""
@bot.message_handler(commands=['change'])
def manipulate_entry(message):
    if message.chat.id not in user_data:
        user_data[message.chat.id] = {}
    try:
        # known vector
        z = user_data[message.chat.id]['latent_vector']
    except Exception:
        # generate new random image/vector
        bot.send_message(message.chat.id, 'No images yet! Generating random..')

        gan_image_path = f'{message.chat.id}-gan-image.png'
        z = model.generate_and_save_random_image(gan_image_path)
        user_data[message.chat.id] = {}
        user_data[message.chat.id]['latent_vector'] = z.cpu().detach().numpy()
        user_data[message.chat.id]['gan_image_path'] = gan_image_path
        user_data[message.chat.id]['original_image_path'] = None
        bot.send_photo(message.chat.id, photo=open(gan_image_path, 'rb'),
                       caption='Your random face! Now send /change again to display'
                               ' possible changes!')
        return

    for prop in model.properties:
        user_data[message.chat.id][prop] = 0
    msg = """
    Select facial attributes to change or preserve:\n
    '\U00002705' - attirbute changes
    '\U0001F512' - attribute stays the same
    otherwise - no guarantees
    """
    bot.send_message(message.chat.id, msg, reply_markup=generate_markup_manipulation_initial(message.chat.id))

"""
Help message.
"""
@bot.message_handler(commands=['help', 'start'])
def send_welcome_help(message):
    helpmessage = """
    Hello! This is Neural Face Editor bot!
    Send /face to generate random faces with GAN!
    Send /change to enter face manipulation menu!
    Send me an image to edit your own one!
    """
    bot.reply_to(message, helpmessage)

"""
Generate random face with GAN.
"""
@bot.message_handler(commands=['face'])
def generate_face(message):
    # new image path, user-specific
    gan_image_path = f'{message.chat.id}-gan-image.png'
    # generate random latent code and generate image
    z = model.generate_and_save_random_image(gan_image_path)
    # fill user data
    user_data[message.chat.id] = {}
    user_data[message.chat.id]['latent_vector'] = z.cpu().detach().numpy()
    user_data[message.chat.id]['gan_image_path'] = gan_image_path
    user_data[message.chat.id]['original_image_path'] = None
    # send user his pic
    bot.send_photo(message.chat.id, photo=open(gan_image_path, 'rb'), caption='Your random face! Now send /change to display'
                                                                              ' possible changes!')


# echo as default
@bot.message_handler(func=lambda x: True)
def echo_all_to_channel(message):
    bot.send_message(message.chat.id, 'Sorry, I don\'t know how to react..')

# start bot
bot.add_custom_filter(custom_filters.ChatFilter())
bot.infinity_polling()
