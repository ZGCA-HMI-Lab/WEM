from dataclasses import dataclass


@dataclass(frozen=True)
class QwenChatTemplateTokens:
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    user_token_id: int = 872
    assistant_token_id: int = 77091
    newline_token_id: int = 198
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    video_token_id: int = 151654
    pad_token_id: int = 0


QWEN_CHAT_TOKENS = QwenChatTemplateTokens()
