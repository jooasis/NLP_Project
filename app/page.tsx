'use client';

import React, { useState, useEffect } from 'react';

// 데이터 구조 정의
interface AnalysisSegment {
  text: string;
  type: 'ORG' | 'PER' | 'LOC' | 'normal';
  confidence: number;
}

interface HistoryItem {
  id: string;
  originalText: string;
  result: AnalysisSegment[];
  timestamp: string;
}

// 범례 항목을 위한 재사용 컴포넌트
const LegendItem = ({ color, label }: { color: string; label: string }) => (
  <div className="flex items-center space-x-2">
    <div className={`w-4 h-4 rounded-full ${color}`}></div>
    <span className="text-sm text-gray-300">{label}</span>
  </div>
);


export default function Page() {
  // --- 상태 관리 ---
  const [userInput, setUserInput] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisSegment[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // --- 효과 (Effects) ---
  useEffect(() => {
    const savedHistory = localStorage.getItem('analysisHistory');
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  // --- 함수 (Functions) ---
  const saveHistory = (newHistoryItem: HistoryItem) => {
    const updatedHistory = [newHistoryItem, ...history].slice(0, 50);
    setHistory(updatedHistory);
    localStorage.setItem('analysisHistory', JSON.stringify(updatedHistory));
  };

  const getHighlightStyle = (type: AnalysisSegment['type'], confidence: number): string => {
    let baseStyle = 'px-2 py-1 rounded-md text-white font-semibold';
    let colorStyle = '';

    switch (type) {
      case 'ORG':
        if (confidence >= 85) colorStyle = 'bg-blue-600';
        else if (confidence >= 70) colorStyle = 'bg-blue-500 bg-opacity-70';
        else colorStyle = 'bg-blue-500 bg-opacity-40';
        break;
      case 'PER':
        if (confidence >= 85) colorStyle = 'bg-green-600';
        else if (confidence >= 70) colorStyle = 'bg-green-500 bg-opacity-70';
        else colorStyle = 'bg-green-500 bg-opacity-40';
        break;
      case 'LOC':
        if (confidence >= 85) colorStyle = 'bg-orange-500';
        else if (confidence >= 70) colorStyle = 'bg-orange-400 bg-opacity-70';
        else colorStyle = 'bg-orange-400 bg-opacity-40';
        break;
      default:
        return 'px-1';
    }
    return `${baseStyle} ${colorStyle}`;
  };

  const handleAnalysis = async (): Promise<void> => {
    if (!userInput.trim()) return;

    setIsLoading(true);
    setAnalysisResult([]);

    await new Promise(resolve => setTimeout(resolve, 1500));

    const mockResponse: AnalysisSegment[] = [
      { text: '최근 ', type: 'normal', confidence: 100 },
      { text: '삼성', type: 'ORG', confidence: 92 },
      { text: '은(는) ', type: 'normal', confidence: 100 },
      { text: '이재용', type: 'PER', confidence: 88 },
      { text: ' 회장의 주도 아래 ', type: 'normal', confidence: 100 },
      { text: '수원', type: 'LOC', confidence: 75 },
      { text: '에서 대규모 투자를 발표했습니다. 저기 ', type: 'normal', confidence: 100 },
      { text: '애플', type: 'ORG', confidence: 65 },
      { text: '은 긴장해야 할 것입니다.', type: 'normal', confidence: 100 },
    ];

    setAnalysisResult(mockResponse);
    setIsLoading(false);

    const newHistoryItem: HistoryItem = {
      id: new Date().toISOString(),
      originalText: userInput,
      result: mockResponse,
      timestamp: new Date().toLocaleString('ko-KR'),
    };
    saveHistory(newHistoryItem);
  };

  const loadFromHistory = (item: HistoryItem) => {
    setUserInput(item.originalText);
    setAnalysisResult(item.result);
  };
  
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setUserInput(e.target.value);
  };

  return (
    <div className="bg-gray-900 h-screen p-4 flex flex-col items-center">
      <div className="flex w-full max-w-5xl gap-8 flex-grow min-h-0">
        
        {/* 왼쪽: 분석기 본체 */}
        <div className="w-2/3 bg-gray-800 rounded-xl shadow-2xl p-8 flex flex-col space-y-6">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-white">애둘러 표현 탐지기</h1>
            <p className="text-gray-400 mt-2">분석하고 싶은 텍스트를 입력하세요.</p>
          </div>
          <textarea
            value={userInput}
            onChange={handleInputChange}
            className="w-full h-32 p-4 bg-gray-700 text-gray-200 rounded-lg border-2 border-gray-600 focus:border-blue-500 focus:ring-0 transition"
            placeholder="예) 요즘 그 회사는 문제야. 거기 대표도 마찬가지고..."
            disabled={isLoading}
          />
          <button
            onClick={handleAnalysis}
            disabled={isLoading || !userInput.trim()}
            className="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
          >
            {isLoading ? ( '분석 중...' ) : ( '분석하기' )}
          </button>
          <div className="flex-grow w-full min-h-[10rem] p-4 bg-gray-700 text-gray-300 rounded-lg border-2 border-gray-600 leading-relaxed overflow-y-auto">
            {analysisResult.length > 0 ? (
              analysisResult.map((segment, index) => (
                <span key={index} className={getHighlightStyle(segment.type, segment.confidence)}>
                  {segment.text}
                </span>
              ))
            ) : (
              <span className="text-gray-500">결과가 여기에 표시됩니다.</span>
            )}
          </div>
        </div>

        {/* 오른쪽: 사이드 패널 (범례 + 기록) */}
        <div className="w-1/3 flex flex-col gap-6">
            <div className="bg-gray-800 rounded-xl shadow-2xl p-6">
                <h3 className="text-lg font-bold text-white mb-4 border-b border-gray-600 pb-2">범례</h3>
                <div className="space-y-3">
                    <LegendItem color="bg-blue-600" label="조직 (ORG)" />
                    <LegendItem color="bg-green-600" label="인물 (PER)" />
                    <LegendItem color="bg-orange-500" label="위치 (LOC)" />
                    <div className="pt-2 mt-2 border-t border-gray-700">
                        <div className="flex items-center space-x-2">
                            <div className="w-4 h-4 rounded-full bg-green-600"></div>
                            <span className="text-sm text-gray-300">신뢰도 85%+</span>
                        </div>
                         <div className="flex items-center space-x-2 mt-2">
                            <div className="w-4 h-4 rounded-full bg-green-500 bg-opacity-70"></div>
                            <span className="text-sm text-gray-300">신뢰도 70-85%</span>
                        </div>
                         <div className="flex items-center space-x-2 mt-2">
                            <div className="w-4 h-4 rounded-full bg-green-500 bg-opacity-40"></div>
                            <span className="text-sm text-gray-300">신뢰도 70% 미만</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* 분석 기록 UI */}
            <div className="bg-gray-800 rounded-xl shadow-2xl p-6 flex flex-col flex-grow min-h-0">
                <div className="flex justify-between items-center mb-4 border-b border-gray-600 pb-2">
                    <h2 className="text-xl font-bold text-white">분석 기록</h2>
                    <button
                        onClick={() => alert('내보내기 기능은 백엔드 연동이 필요합니다.')}
                        className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition"
                        title="기록 내보내기"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-5 h-5">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 8.25H7.5a2.25 2.25 0 0 0-2.25 2.25v9a2.25 2.25 0 0 0 2.25 2.25h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25H15m0-3-3-3m0 0-3 3m3-3V15" />
                        </svg>
                    </button>
                </div>
                
                {/* ★★★ 여기가 수정된 부분입니다 ★★★ */}
                {/* pr-2 클래스를 추가하여 스크롤바 공간을 확보합니다. */}
                <div className="overflow-y-auto flex-grow min-h-0 pr-4">
                    {history.length > 0 ? (
                    <ul className="space-y-3">
                        {history.map((item) => (
                        <li key={item.id}>
                            <button
                            onClick={() => loadFromHistory(item)}
                            className="w-full text-left p-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition"
                            >
                            <p className="text-sm text-white truncate">{item.originalText}</p>
                            <p className="text-xs text-gray-400 mt-1">{item.timestamp}</p>
                            </button>
                        </li>
                        ))}
                    </ul>
                    ) : (
                    <p className="text-gray-500 text-center mt-8">분석 기록이 없습니다.</p>
                    )}
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}