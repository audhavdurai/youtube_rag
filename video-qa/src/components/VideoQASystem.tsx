"use client";

import {
  Loader2,
  Send,
  Upload,
  AlertCircle,
  Image as ImageIcon,
  Plus,
  Menu,
  X,
  Video,
  Info,
  Pencil,
  LogOut,
} from "lucide-react";
import React, { useState, useEffect, useRef, useContext } from "react";
import UsernameModal from "./login";

interface VideoProcessResponse {
  status: string;
  video_info: any;
  frames_stored: number;
  transcript_chunks: number;
}

interface SearchResult {
  id: string;
  title: string;
  description: string | null;
  thumbnail: string;
  channelTitle: string;
  url: string;
}

interface ChatMessage {
  type: "question" | "answer";
  content: string;
  context?: {
    images?: string[];
    texts?: string[];
  };
  searchResults?: SearchResult[]; // Add this for search results
  timestamp: number;
}

interface QueryResponse {
  answer: string;
  context: {
    images: string[];
    texts: string[];
  };
}

interface VideoInfo {
  title: string;
  url: string;
  timestamp: number;
}

interface Chat {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: number;
  updated_at: number;
  videos: VideoInfo[]; // Add this
}
const ngrokurl = "https://dcbe-108-51-25-37.ngrok-free.app/";

const SearchResultsList = React.memo(
  ({ results }: { results: SearchResult[] | undefined }) => {
    if (!results) return null;

    return (
      <div className="grid grid-cols-1 gap-4 mt-3">
        {results.map((video) => (
          <SearchResultItem key={video.id} video={video} />
        ))}
      </div>
    );
  }
);

// 2. Extract individual search result item as a separate component
const SearchResultItem = React.memo(({ video }: { video: SearchResult }) => {
  const { setVideoUrl, setShowModal } = useContext(VideoContext);

  return (
    <div className="bg-[#090b10] rounded-lg overflow-hidden flex flex-col">
      <div className="aspect-w-16 aspect-h-9">
        <iframe
          src={`https://www.youtube.com/embed/${video.id}`}
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          className="w-full h-full"
        />
      </div>
      <div className="p-3">
        <h3 className="text-sm font-medium text-gray-200">{video.title}</h3>
        <p className="text-xs text-gray-400 mt-1">{video.channelTitle}</p>
        {video.description && (
          <p className="text-xs text-gray-500 mt-1 line-clamp-2">
            {video.description}
          </p>
        )}
        <button
          onClick={() => {
            setVideoUrl(video.url);
            setShowModal(true);
          }}
          className="mt-2 ml-auto text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700 transition-colors flex items-center gap-1"
        >
          <Upload size={12} />
          Process
        </button>
      </div>
    </div>
  );
});

// 3. Create a context to avoid prop drilling
interface VideoContextType {
  setVideoUrl: (url: string) => void;
  setShowModal: (show: boolean) => void;
}

const VideoContext = React.createContext<VideoContextType>({
  setVideoUrl: () => {},
  setShowModal: () => {},
});

const VideoQASystem: React.FC = () => {
  // Chat management
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [showVideoSidebar, setShowVideoSidebar] = useState(false);
  const [frameInterval, setFrameInterval] = useState(1); // Default to 60 seconds

  // UI states
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [question, setQuestion] = useState<string>("");
  const [processing, setProcessing] = useState<boolean>(false);
  const [querying, setQuerying] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [showModal, setShowModal] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [openContexts, setOpenContexts] = useState<{ [key: number]: boolean }>(
    {}
  );
  const [isMounted, setIsMounted] = useState(false);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");

  //   const [username, setUsername] = useState<string | null>(null);
  //   const [showUsernameModal, setShowUsernameModal] = useState(true);
  const [username, setUsername] = useState<string | null>(null);
  const [showUsernameModal, setShowUsernameModal] = useState(true); 

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentChatId, chats]);

  useEffect(() => {
    const fetchChats = async () => {
      try {
        console.log(username);
        const response = await fetch(
          //`http://127.0.0.1:5000/api/chats?username=${username}`
          `${ngrokurl}/api/chats?username=${username}`
        );
        if (!response.ok) {
          throw new Error("Failed to fetch chats");
        }

        const fetchedChats = await response.json();

        const formattedChats: Chat[] = fetchedChats
          .map((chat: any) => {
            // Format messages to handle both regular and search queries
            const messages = (chat.all_messages || []).map((msg: any) => {
              if (msg.type === "answer" && Array.isArray(msg.content)) {
                // This is a search result message
                return {
                  type: "answer",
                  content: "Here are some relevant videos I found:",
                  searchResults: msg.content.map((video: any) => ({
                    id: video.id,
                    title: video.title,
                    description: video.description,
                    thumbnail: video.thumbnail,
                    channelTitle: video.channelTitle,
                    url: video.url,
                  })),
                  timestamp: new Date(msg.timestamp).getTime(),
                };
              } else {
                // Regular message
                return {
                  ...msg,
                  timestamp: new Date(msg.timestamp).getTime(),
                };
              }
            });

            return {
              id: chat.id,
              title: chat.title,
              messages: messages,
              created_at: new Date(chat.created_at).getTime(),
              updated_at: new Date(chat.updated_at).getTime(),
              videos: chat.videos || [],
            };
          })
          .sort(
            (a: { updated_at: number }, b: { updated_at: number }) =>
              b.updated_at - a.updated_at
          );

        setChats(formattedChats);
        if (formattedChats.length > 0) {
          setCurrentChatId(formattedChats[0].id);
        }
      } catch (error) {
        console.error("Error fetching chats:", error);
        setError("Failed to load chats");
      }
    };

    fetchChats();
  }, [username]);

  const createNewChat = async () => {
    try {
      const response = await fetch(`${ngrokurl}/api/chats`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: "New Chat",
          username: username,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to create chat");
      }

      const newChat = await response.json();

      const chat: Chat = {
        ...newChat,
        created_at: new Date(newChat.created_at).getTime(),
        updated_at: new Date(newChat.updated_at).getTime(),
        messages: [],
      };

      setChats((prev) => [chat, ...prev]);
      setCurrentChatId(chat.id);
    } catch (error) {
      setError("Failed to create new chat");
      console.error("Error creating chat:", error);
    }
  };

  const handleUsernameSubmit = async (newUsername: string) => {
    setUsername(newUsername);
    sessionStorage.setItem("username", newUsername);
    setShowUsernameModal(false);
  };

  const updateChatTitle = async (chatId: string, newTitle: string) => {
    try {
      const response = await fetch(
        `${ngrokurl}/api/chats/${chatId}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: newTitle,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to update chat title");
      }

      setChats((prevChats) =>
        prevChats.map((chat) =>
          chat.id === chatId ? { ...chat, title: newTitle } : chat
        )
      );
      setEditingChatId(null);
    } catch (error) {
      console.error("Error updating chat title:", error);
      setError("Failed to update chat title");
    }
  };

  const deleteChat = async (chatId: string) => {
    try {
      const response = await fetch(
        `${ngrokurl}/api/chats/${chatId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to delete chat");
      }

      setChats((prev) => prev.filter((chat) => chat.id !== chatId));
      if (currentChatId === chatId) {
        setCurrentChatId(chats[0]?.id || null);
      }
    } catch (error) {
      setError("Failed to delete chat");
      console.error("Error deleting chat:", error);
    }
  };

  const getCurrentChat = () => chats.find((chat) => chat.id === currentChatId);

  const getYouTubeEmbedUrl = (url: string) => {
    const regExp =
      /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return match && match[2].length === 11
      ? `https://www.youtube.com/embed/${match[2]}`
      : undefined;
  };

  const processVideo = async (): Promise<void> => {
    if (!videoUrl || !currentChatId) {
      setError("Please enter a YouTube URL");
      return;
    }

    setProcessing(true);
    setError("");

    try {
      const response = await fetch(`${ngrokurl}/api/process-video`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: videoUrl,
          chat_id: currentChatId,
          frame_interval: frameInterval,
        }),
      });

      const data: VideoProcessResponse = await response.json();

      if (!response.ok) {
        throw new Error(data.status || "Failed to process video");
      }

      setChats((prevChats) =>
        prevChats.map((chat) => {
          if (chat.id === currentChatId) {
            const newVideo: VideoInfo = {
              title: data.video_info.title,
              url: videoUrl,
              timestamp: Date.now(),
            };

            return {
              ...chat,
              videos: [...(chat.videos || []), newVideo],
              updated_at: Date.now(),
            };
          }
          return chat;
        })
      );

      setError("");
      setShowModal(false);
      setVideoUrl("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setProcessing(false);
    }
  };

  const askQuestion = async (): Promise<void> => {
    if (!question.trim() || !currentChatId) {
      return;
    }

    setQuerying(true);
    setError("");

    try {
      const currentChat = getCurrentChat();
      if (!currentChat) return;

      // Create new question
      const newQuestion: ChatMessage = {
        type: "question",
        content: question,
        timestamp: Date.now(),
      };

      // Add question to messages
      const messagesWithQuestion = [...currentChat.messages, newQuestion];
      setChats((prevChats) =>
        prevChats.map((chat) =>
          chat.id === currentChatId
            ? {
                ...chat,
                messages: messagesWithQuestion,
                updated_at: Date.now(),
              }
            : chat
        )
      );

      setQuestion("");

      // Make API call
      const response = await fetch(`${ngrokurl}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          chat_id: currentChatId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error("Failed to get answer");
      }

      // Handle search results
      if (data.type === "search") {
        const searchAnswer: ChatMessage = {
          type: "answer",
          content: "Here are some relevant videos I found:",
          searchResults: data.results,
          timestamp: Date.now(),
        };

        setChats((prevChats) =>
          prevChats.map((chat) =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  messages: [...messagesWithQuestion, searchAnswer],
                  updated_at: Date.now(),
                }
              : chat
          )
        );
      } else {
        // Handle regular query response
        const newAnswer: ChatMessage = {
          type: "answer",
          content: data.results.answer,
          context: data.results.context,
          timestamp: Date.now(),
        };

        setChats((prevChats) =>
          prevChats.map((chat) =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  messages: [...messagesWithQuestion, newAnswer],
                  updated_at: Date.now(),
                }
              : chat
          )
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setQuerying(false);
    }
  };

  const Modal = () => {
    const modalInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
      if (showModal && modalInputRef.current) {
        setTimeout(() => {
          modalInputRef.current?.focus();
        }, 50);
      }
    }, [showModal]);

    if (!showModal) return null;

    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-[#1a1d27] rounded-lg p-6 w-96">
          <h3 className="text-lg font-medium text-gray-200 mb-4">
            Enter YouTube URL
          </h3>
          <input
            type="text"
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
            placeholder="https://youtube.com/..."
            className="w-full p-3 bg-[#090b10] text-gray-200 placeholder-gray-500 rounded-lg border border-gray-700 mb-4"
            onKeyDown={(e) => {
              if (e.key === "Enter") processVideo();
              if (e.key === "Escape") setShowModal(false);
            }}
          />

          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowModal(false)}
              className="px-4 py-2 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg"
            >
              Cancel
            </button>
            <button
              onClick={processVideo}
              disabled={processing}
              className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-blue-800 disabled:text-gray-300"
            >
              {processing ? (
                <Loader2 className="animate-spin" size={18} />
              ) : (
                <Upload size={18} />
              )}
              Process
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-[#0f1116] flex">
      {/* Left Sidebar */}
      <div
        className={`fixed md:relative w-72 h-full bg-[#090b10] transform transition-transform duration-200 ease-in-out ${
          showSidebar ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        } flex flex-col z-20`}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between h-14 px-4 border-b border-gray-800">
          <h2 className="text-lg font-medium text-gray-200">Conversations</h2>
          <button
            onClick={createNewChat}
            className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-md transition-colors"
          >
            <Plus size={20} />
          </button>
        </div>

        {/* Conversations List */}
        {chats.map((chat) => (
          <div key={chat.id} className="group border-b border-gray-800/50">
            <div
              onClick={() => setCurrentChatId(chat.id)}
              className="px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 cursor-pointer"
            >
              <div className="min-w-0 flex-1">
                {editingChatId === chat.id ? (
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        updateChatTitle(chat.id, editTitle);
                      }
                      if (e.key === "Escape") {
                        setEditingChatId(null);
                      }
                    }}
                    onBlur={() => {
                      if (editTitle.trim()) {
                        updateChatTitle(chat.id, editTitle);
                      } else {
                        setEditingChatId(null);
                      }
                    }}
                    className="w-full bg-[#1a1d27] text-sm font-medium text-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <>
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-medium text-gray-300 truncate">
                        {chat.title}
                      </h3>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingChatId(chat.id);
                          setEditTitle(chat.title);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-gray-300 transition-opacity"
                      >
                        <Pencil size={12} />
                      </button>
                    </div>
                    <p className="text-xs text-gray-500 truncate mt-0.5">
                      {chat.messages[chat.messages.length - 1]?.content ||
                        "No messages"}
                    </p>
                  </>
                )}
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteChat(chat.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-gray-300 transition-opacity"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        ))}

        {/* User Footer - Add this new section */}
        <div className="mt-auto">
          <div className="px-4 py-3 flex items-center justify-between">
            <div className="flex items-center">
              <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white font-medium">
                {username?.[0]?.toUpperCase()}
              </div>
              <span className="ml-2 text-sm text-gray-300">{username}</span>
            </div>
            <button
              onClick={() => {
                sessionStorage.removeItem("username");
                setUsername(null);
                setShowUsernameModal(true);
                setChats([]);
                setCurrentChatId(null);
              }}
              className="p-2 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-colors"
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div
        className={`flex-1 flex flex-col min-w-0 transition-all duration-200 ease-in-out ${
          showVideoSidebar ? "md:mr-80" : "md:mr-0"
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between h-14 px-4 border-b border-gray-800 bg-[#090b10]">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="md:hidden text-gray-400 hover:text-gray-200"
            >
              <Menu size={20} />
            </button>
            <h1 className="text-lg font-medium text-gray-200">
              {getCurrentChat()?.title || "New Chat"}
            </h1>
          </div>
          <button
            onClick={() => setShowVideoSidebar(!showVideoSidebar)}
            className="text-gray-400 hover:text-gray-200"
          >
            <Video size={20} />
          </button>
        </div>
        <VideoContext.Provider value={{ setVideoUrl, setShowModal }}>
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[#0f1116]">
            {getCurrentChat()?.messages.map((message, index) => {
              const hasContext =
                message.type === "answer" &&
                message.context?.images &&
                message.context?.images.length > 0;
              const hasSearchResults =
                message.type === "answer" &&
                message.searchResults &&
                message.searchResults.length > 0;

              return (
                <div
                  key={index}
                  className={`flex ${
                    message.type === "question"
                      ? "justify-end"
                      : "justify-start"
                  } relative`}
                >
                  <div
                    className={`relative max-w-[80%] p-3 rounded-lg ${
                      message.type === "question"
                        ? "bg-blue-600 text-white"
                        : "bg-[#1a1d27] text-gray-200"
                    }`}
                  >
                    <p className="text-sm leading-relaxed">{message.content}</p>

                    {hasSearchResults && (
                      <SearchResultsList results={message.searchResults} />
                    )}

                    {hasContext && (
                      <button
                        onClick={() => {
                          setOpenContexts((prev) => ({
                            ...prev,
                            [index]: !prev[index],
                          }));
                        }}
                        className="text-gray-400 hover:text-gray-200 p-1 rounded-full hover:bg-black/20 transition-colors mt-[-4px]"
                      >
                        <Info size={16} />
                      </button>
                    )}
                  </div>

                  {/* Context popup remains the same */}
                  {isMounted && openContexts[index] && (
                    <div className="absolute left-0 top-full mt-2 w-80 bg-[#1a1d27] rounded-lg shadow-lg border border-gray-800 z-30">
                      <div className="flex items-center justify-between p-3 border-b border-gray-800">
                        <h3 className="text-sm font-medium text-gray-200">
                          Context
                        </h3>
                        <button
                          onClick={() => {
                            setOpenContexts((prev) => ({
                              ...prev,
                              [index]: false,
                            }));
                          }}
                          className="text-gray-400 hover:text-gray-200"
                        >
                          <X size={16} />
                        </button>
                      </div>
                      <div className="p-3 max-h-96 overflow-y-auto">
                        {message.context?.images &&
                          message.context?.images.length > 0 && (
                            <div className="grid grid-cols-2 gap-2">
                              {message.context?.images.map(
                                (image, imgIndex) => (
                                  <img
                                    key={imgIndex}
                                    src={`data:image/jpeg;base64,${image}`}
                                    alt={`Context image ${imgIndex + 1}`}
                                    className="w-full h-32 object-cover rounded-lg"
                                  />
                                )
                              )}
                            </div>
                          )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-800 bg-[#090b10]">
            <div className="flex gap-3 items-center">
              <div className="flex-1 bg-[#1a1d27] rounded-lg">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      askQuestion();
                    }
                  }}
                  placeholder="Ask about the video..."
                  className="w-full bg-transparent text-gray-200 placeholder-gray-500 p-3 resize-none focus:outline-none text-sm min-h-[44px]"
                  rows={1}
                  disabled={!currentChatId}
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowModal(true)}
                  disabled={!currentChatId}
                  className="p-2 text-gray-400 hover:text-gray-200 bg-[#1a1d27] rounded-lg transition-colors disabled:opacity-50"
                >
                  <Upload size={20} />
                </button>
                <button
                  onClick={askQuestion}
                  disabled={!currentChatId || !question.trim()}
                  className="p-2 text-gray-400 hover:text-gray-200 bg-[#1a1d27] rounded-lg transition-colors disabled:opacity-50"
                >
                  {querying ? (
                    <Loader2 className="animate-spin" size={20} />
                  ) : (
                    <Send size={20} />
                  )}
                </button>
              </div>
            </div>
            {error && (
              <div className="mt-2 text-red-400 text-sm flex items-center gap-2">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}
          </div>

          <Modal />
        </VideoContext.Provider>

        {showUsernameModal && <UsernameModal onSubmit={handleUsernameSubmit} />}
      </div>

      <div
        className={`fixed w-80 h-full bg-[#090b10] transform transition-all duration-200 ease-in-out ${
          showVideoSidebar ? "translate-x-0" : "translate-x-full"
        } flex flex-col border-l border-gray-800 z-10 right-0 top-0`}
      >
        <div className="flex items-center justify-between h-14 px-4 border-b border-gray-800">
          <h2 className="text-lg font-medium text-gray-200">Videos</h2>
          <button
            onClick={() => setShowVideoSidebar(false)}
            className="md:hidden p-1.5 text-gray-400 hover:text-gray-200"
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {!getCurrentChat()?.videos ||
          getCurrentChat()?.videos.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 space-y-3">
              <Video size={48} className="opacity-20" />
              <p className="text-sm">No videos added yet</p>
              <p className="text-xs max-w-[200px]">
                Upload a YouTube video to start asking questions about it
              </p>
            </div>
          ) : (
            getCurrentChat()?.videos?.map((video, index) => {
              const embedUrl = getYouTubeEmbedUrl(video.url);
              if (!embedUrl) return null;

              return (
                <div
                  key={index}
                  className="bg-[#1a1d27] rounded-lg overflow-hidden"
                >
                  <div className="aspect-w-16 aspect-h-9">
                    <iframe
                      src={embedUrl}
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                      className="w-full h-full"
                    />
                  </div>
                  <div className="p-3">
                    <h3 className="text-sm font-medium text-gray-200 truncate">
                      {video.title}
                    </h3>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(video.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
export default VideoQASystem;
