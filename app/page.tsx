'use client';

import React, { useState, useEffect, useRef } from 'react';

// --- 데이터 구조 정의 ---
interface AnalysisSegment {
  text: string;
  type: 'ORG' | 'PER' | 'LOC' | 'normal';
  confidence: number;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string | AnalysisSegment[];
}

interface HistoryItem {
  id: string;
  originalText: string;
  result?: AnalysisSegment[]; // 분석기 모드용 결과 저장
  timestamp: string;
}

// --- 범례 아이템 컴포넌트 ---
const LegendItem = ({ color, label }: { color: string; label: string }) => (
  <div className="flex items-center space-x-2">
    <div className={`w-4 h-4 rounded-full ${color}`}></div>
    <span className="text-sm text-gray-300">{label}</span>
  </div>
);

export default function Page() {
  // ★★★ 모드 상태 관리 (analyzer | chat) ★★★
  const [viewMode, setViewMode] = useState<'analyzer' | 'chat'>('analyzer');

  // --- 공통 상태 ---
  const [userInput, setUserInput] = useState<string>('');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [mobileTab, setMobileTab] = useState<'main' | 'history'>('main'); // 모바일용 탭

  // --- 분석기 모드용 상태 ---
  const [analysisResult, setAnalysisResult] = useState<AnalysisSegment[]>([]);
  const [isAnalyzerLoading, setIsAnalyzerLoading] = useState<boolean>(false);
  const analyzerResultRef = useRef<HTMLDivElement>(null);

  // --- 채팅 모드용 상태 ---
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingResponse, setStreamingResponse] = useState<AnalysisSegment[]>([]);
  const [isChatStreaming, setIsChatStreaming] = useState<boolean>(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // --- 효과 (Effects) ---
  useEffect(() => {
    const savedHistory = localStorage.getItem('unifiedHistory');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  // 스크롤 자동 이동 (분석기)
  useEffect(() => {
    if (analyzerResultRef.current) {
      analyzerResultRef.current.scrollTop = analyzerResultRef.current.scrollHeight;
    }
  }, [analysisResult]);

  // 스크롤 자동 이동 (채팅)
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, streamingResponse, isChatStreaming]);

  // --- 공통 함수 ---
  const saveHistory = (text: string, result?: AnalysisSegment[]) => {
    const newLog: HistoryItem = {
      id: new Date().toISOString(),
      originalText: text,
      result: result, // 분석기 모드일 경우 결과도 저장
      timestamp: new Date().toLocaleString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
    };
    const updatedHistory = [newLog, ...history].slice(0, 50);
    setHistory(updatedHistory);
    localStorage.setItem('unifiedHistory', JSON.stringify(updatedHistory));
  };

  const getHighlightStyle = (type: AnalysisSegment['type'], confidence: number): string => {
    let baseStyle = 'px-1 rounded text-white font-semibold transition-colors duration-300';
    let colorStyle = '';

    switch (type) {
      case 'ORG':
        if (confidence >= 85) colorStyle = 'bg-blue-600';
        else if (confidence >= 70) colorStyle = 'bg-blue-500 bg-opacity-80';
        else colorStyle = 'bg-blue-500 bg-opacity-50';
        break;
      case 'PER':
        if (confidence >= 85) colorStyle = 'bg-green-600';
        else if (confidence >= 70) colorStyle = 'bg-green-500 bg-opacity-80';
        else colorStyle = 'bg-green-500 bg-opacity-50';
        break;
      case 'LOC':
        if (confidence >= 85) colorStyle = 'bg-orange-500';
        else if (confidence >= 70) colorStyle = 'bg-orange-400 bg-opacity-80';
        else colorStyle = 'bg-orange-400 bg-opacity-50';
        break;
      default:
        return '';
    }
    return `${baseStyle} ${colorStyle}`;
  };

  // --- 가짜 데이터 생성기 ---
  const generateMockResponse = (text: string): AnalysisSegment[] => {
    return [
      { text: '입력하신내용 ', type: 'normal', confidence: 100 },
      { text: '삼성', type: 'ORG', confidence: 92 },
      { text: '과 ', type: 'normal', confidence: 100 },
      { text: '이재용', type: 'PER', confidence: 88 },
      { text: ' 회장이 ', type: 'normal', confidence: 100 },
      { text: '수원', type: 'LOC', confidence: 75 },
      { text: '에서 언급되었습니다.', type: 'normal', confidence: 100 },
    ];
  };

  // --- 분석기 모드 로직 ---
  const handleAnalyzerRun = async () => {
    if (!userInput.trim()) return;
    setIsAnalyzerLoading(true);
    setAnalysisResult([]);
    
    const response = generateMockResponse(userInput);

    // 스트리밍 시뮬레이션
    for (let i = 0; i < response.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 50));
      setAnalysisResult(prev => [...prev, response[i]]);
    }

    saveHistory(userInput, response);
    setIsAnalyzerLoading(false);
  };

  // --- 채팅 모드 로직 ---
  const handleChatSend = async () => {
    if (!userInput.trim() || isChatStreaming) return;
    const text = userInput;
    setUserInput('');
    
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    saveHistory(text);

    setIsChatStreaming(true);
    setStreamingResponse([]);

    const response = generateMockResponse(text);

    // 스트리밍 시뮬레이션
    for (let i = 0; i < response.length; i++) {
      const delay = Math.floor(Math.random() * 80) + 30;
      await new Promise(resolve => setTimeout(resolve, delay));
      setStreamingResponse(prev => [...prev, response[i]]);
    }

    setMessages(prev => [...prev, { role: 'assistant', content: response }]);
    setStreamingResponse([]);
    setIsChatStreaming(false);
  };

  const loadFromHistory = (item: HistoryItem) => {
    if (viewMode === 'analyzer') {
      setUserInput(item.originalText);
      if (item.result) setAnalysisResult(item.result);
    } else {
      setUserInput(item.originalText);
    }
    setMobileTab('main');
  };

  return (
    <div className="bg-gray-900 h-screen flex flex-col items-center overflow-hidden">
      
      {/* ★★★ 헤더 & 모드 스위처 ★★★ */}
      <header className="w-full bg-gray-800 p-3 shadow-md z-10 shrink-0 border-b border-gray-700">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-3">
             <h1 className="text-xl font-bold text-white flex items-center gap-2">
                <span className="bg-blue-600 w-8 h-8 rounded-lg flex items-center justify-center text-sm">AI</span>
                애둘러 표현 탐지기
             </h1>
             
             {/* 모드 전환 토글 버튼 */}
             <div className="flex bg-gray-900 p-1 rounded-lg border border-gray-700">
                <button
                    onClick={() => setViewMode('analyzer')}
                    className={`px-4 py-2 rounded-md text-sm font-bold transition-all ${
                        viewMode === 'analyzer' ? 'bg-gray-700 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
                    }`}
                >
                    📝 분석기 모드
                </button>
                <button
                    onClick={() => setViewMode('chat')}
                    className={`px-4 py-2 rounded-md text-sm font-bold transition-all ${
                        viewMode === 'chat' ? 'bg-blue-600 text-white shadow-sm' : 'text-gray-400 hover:text-gray-200'
                    }`}
                >
                    💬 채팅 모드
                </button>
             </div>

             {/* 모바일 탭 (기록 보기용) */}
             <button
                className="lg:hidden text-gray-400 text-sm"
                onClick={() => setMobileTab(mobileTab === 'main' ? 'history' : 'main')}
             >
                {mobileTab === 'main' ? '기록 보기 >' : '< 메인으로'}
             </button>
        </div>
      </header>

      <div className="flex w-full max-w-6xl flex-grow min-h-0 relative">
        
        {/* ========================================================== */}
        {/* ★★★ 메인 영역 (분석기 OR 채팅) ★★★ */}
        {/* ========================================================== */}
        <div className={`
            flex-col flex-grow relative bg-gray-900 transition-all duration-300
            ${mobileTab === 'main' ? 'flex' : 'hidden lg:flex'}
        `}>
            
            {/* --- 1. 분석기 모드 UI --- */}
            {viewMode === 'analyzer' && (
                <div className="h-full flex flex-col p-6 max-w-4xl mx-auto w-full gap-6">
                    <div className="text-center shrink-0">
                        <p className="text-gray-400">긴 텍스트를 한 번에 분석할 때 유용합니다.</p>
                    </div>
                    <textarea
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        className="w-full h-40 p-4 bg-gray-800 text-gray-200 rounded-xl border border-gray-600 focus:border-blue-500 focus:ring-0 transition resize-none text-lg"
                        placeholder="분석할 텍스트를 입력하세요..."
                        disabled={isAnalyzerLoading}
                    />
                    <button
                        onClick={handleAnalyzerRun}
                        disabled={isAnalyzerLoading || !userInput.trim()}
                        className="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl transition shadow-lg shrink-0"
                    >
                        {isAnalyzerLoading ? '분석 중...' : '분석 시작'}
                    </button>
                    
                    {/* 결과 박스 */}
                    <div 
                        ref={analyzerResultRef}
                        className="flex-grow w-full bg-gray-800 p-6 rounded-xl border border-gray-700 overflow-y-auto leading-relaxed text-lg text-gray-300 shadow-inner"
                    >
                        {analysisResult.length > 0 ? (
                            analysisResult.map((segment, index) => (
                                <span key={index} className={getHighlightStyle(segment.type, segment.confidence)}>
                                    {segment.text}
                                </span>
                            ))
                        ) : (
                            <div className="h-full flex items-center justify-center text-gray-500 flex-col gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10 opacity-50">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                                </svg>
                                <span>결과가 여기에 표시됩니다.</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* --- 2. 채팅 모드 UI --- */}
            {viewMode === 'chat' && (
                <div className="flex flex-col h-full w-full">
                    {/* 채팅 리스트 */}
                    <div ref={chatContainerRef} className="flex-grow overflow-y-auto p-4 space-y-6 pb-4 scroll-smooth">
                        {messages.length === 0 && (
                            <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-2 opacity-50">
                                <p>대화하듯 자연스럽게 분석해보세요.</p>
                            </div>
                        )}
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[85%] md:max-w-[75%] p-4 rounded-2xl shadow-md text-base leading-relaxed ${
                                    msg.role === 'user' 
                                    ? 'bg-blue-600 text-white rounded-tr-none' 
                                    : 'bg-gray-800 text-gray-200 rounded-tl-none border border-gray-700'
                                }`}>
                                    {msg.role === 'user' ? (
                                        <p className="whitespace-pre-wrap">{msg.content as string}</p>
                                    ) : (
                                        <div>
                                            {(msg.content as AnalysisSegment[]).map((seg, i) => (
                                                <span key={i} className={getHighlightStyle(seg.type, seg.confidence)}>{seg.text}</span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                        {/* 스트리밍 중 표시 */}
                        {isChatStreaming && (
                            <div className="flex justify-start">
                                <div className="max-w-[85%] md:max-w-[75%] p-4 rounded-2xl rounded-tl-none bg-gray-800 text-gray-200 border border-gray-700 shadow-md">
                                    {streamingResponse.map((seg, i) => (
                                        <span key={i} className={getHighlightStyle(seg.type, seg.confidence)}>{seg.text}</span>
                                    ))}
                                    <span className="inline-block w-2 h-5 ml-1 align-middle bg-blue-500 animate-pulse"></span>
                                </div>
                            </div>
                        )}
                    </div>
                    {/* 채팅 입력바 */}
                    <div className="p-4 bg-gray-900 border-t border-gray-800">
                        <div className="max-w-4xl mx-auto flex items-end gap-2 bg-gray-800 p-2 rounded-xl border border-gray-700 focus-within:border-blue-500 transition-colors">
                            <textarea
                                value={userInput}
                                onChange={(e) => setUserInput(e.target.value)}
                                onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChatSend(); }}}
                                placeholder="메시지를 입력하세요..."
                                className="w-full bg-transparent text-white p-2 max-h-32 min-h-[50px] resize-none focus:outline-none"
                                rows={1}
                                disabled={isChatStreaming}
                            />
                            <button
                                onClick={handleChatSend}
                                disabled={!userInput.trim() || isChatStreaming}
                                className="p-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shrink-0 mb-[1px]"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                                    <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>

        {/* ========================================================== */}
        {/* ★★★ 오른쪽 사이드 패널 (공통) ★★★ */}
        {/* ========================================================== */}
        <div className={`
            flex-col gap-4 border-l border-gray-800 bg-gray-900 p-4 w-full
            lg:w-80 lg:flex lg:static lg:h-auto
            ${mobileTab === 'history' ? 'absolute inset-0 z-20 flex' : 'hidden'}
        `}>
            {/* 범례 */}
            <div className="bg-gray-800 rounded-xl shadow-lg p-5 shrink-0">
                <h3 className="text-sm font-bold text-gray-400 mb-3 uppercase tracking-wider">Highlight Legend</h3>
                <div className="space-y-3">
                    <LegendItem color="bg-blue-600" label="조직 (ORG)" />
                    <LegendItem color="bg-green-600" label="인물 (PER)" />
                    <LegendItem color="bg-orange-500" label="위치 (LOC)" />
                    <div className="pt-2 mt-2 border-t border-gray-700">
                        <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 rounded-full bg-green-600"></div>
                            <span className="text-xs text-gray-400">85%+ (확실)</span>
                        </div>
                         <div className="flex items-center space-x-2 mt-2">
                            <div className="w-3 h-3 rounded-full bg-green-500 bg-opacity-70"></div>
                            <span className="text-xs text-gray-400">70-85% (보통)</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* 통합 기록 */}
            <div className="bg-gray-800 rounded-xl shadow-lg p-5 flex flex-col flex-grow min-h-0">
                <div className="flex justify-between items-center mb-3">
                    <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">History</h3>
                    <button title="내보내기" onClick={() => alert('백엔드 연동 필요')} className="text-gray-500 hover:text-white">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                        </svg>
                    </button>
                </div>
                <div className="overflow-y-auto flex-grow pr-2 space-y-2">
                    {history.length > 0 ? (
                        history.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => loadFromHistory(item)}
                            className="w-full text-left p-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition group"
                        >
                            <p className="text-sm text-gray-200 truncate group-hover:text-white">{item.originalText}</p>
                            <p className="text-xs text-gray-500 mt-1">{item.timestamp}</p>
                        </button>
                        ))
                    ) : (
                        <p className="text-gray-500 text-xs text-center py-4">기록이 없습니다.</p>
                    )}
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}