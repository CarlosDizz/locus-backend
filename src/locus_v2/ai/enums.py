from enum import StrEnum


class ServiceKind(StrEnum):
    CHAT = "chat"
    VOICE = "voice"


class Lifecycle(StrEnum):
    STABLE = "stable"
    PREVIEW = "preview"
    DISABLED = "disabled"
    RETIRED = "retired"


class PublicationStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    RETIRED = "retired"


class VoiceMode(StrEnum):
    PUSH_TO_TALK = "push_to_talk"
    CONTINUOUS = "continuous"
