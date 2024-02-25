import base64

def img_to_b64(img_path):
    file_ = open(img_path, "rb")
    contents = file_.read()
    b64_data = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return b64_data

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{img_to_b64("resources/android.png")}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">[[MSG]]</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{img_to_b64("resources/human.png")}" alt="human" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;" />
    </div>    
    <div class="message">[[MSG]]</div>
</div>
'''
