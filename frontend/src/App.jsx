import React, { useState, useEffect } from 'react';
import { Search, Database, Beaker, BookOpen, Upload, ChevronDown, ChevronUp, AlertCircle, CheckCircle, Info, Loader2 } from 'lucide-react';

const API_BASE = '/api';

const BactAIApp = () => {
  const [activeTab, setActiveTab] = useState('identify');
  const [phenotypeText, setPhenotypeText] = useState('');
  const [useLLM, setUseLLM] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [expandedGenus, setExpandedGenus] = useState(0);
  const [trainingOutput, setTrainingOutput] = useState(null);
  const [dbLastUpdated, setDbLastUpdated] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check backend health on mount
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => {
        setDbLastUpdated(data.db_last_updated || 'Unknown');
      })
      .catch(err => {
        console.error('Backend health check failed:', err);
        setError('Cannot connect to backend. Make sure Flask server is running on port 8000.');
      });
  }, []);

  const analyzeText = async () => {
    if (!phenotypeText.trim()) {
      alert('Please enter a phenotype description');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      const response = await fetch(`${API_BASE}/identify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: phenotypeText,
          use_llm: useLLM
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResults(data);
      setExpandedGenus(0);
    } catch (error) {
      console.error('Error analyzing text:', error);
      setError('Failed to analyze text: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const runTraining = async (endpoint, buttonLabel) => {
    if (!confirm(`Are you sure you want to run: ${buttonLabel}?`)) {
      return;
    }

    setLoading(true);
    setTrainingOutput(null);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/${endpoint}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setTrainingOutput(data);
    } catch (error) {
      console.error('Training error:', error);
      setTrainingOutput({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence === 'Excellent Identification') return 'text-green-400';
    if (confidence === 'Good Identification') return 'text-yellow-400';
    if (confidence === 'Acceptable Identification') return 'text-orange-400';
    return 'text-red-400';
  };

  const getConfidenceEmoji = (confidence) => {
    if (confidence === 'Excellent Identification') return '🟢';
    if (confidence === 'Good Identification') return '🟡';
    if (confidence === 'Acceptable Identification') return '🟠';
    return '🔴';
  };

  const getProgressBarColor = (percent) => {
    if (percent >= 80) return 'bg-blue-500';
    if (percent >= 65) return 'bg-green-500';
    if (percent >= 50) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const renderIdentification = () => (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
        <label className="block text-gray-200 text-sm font-medium mb-2">
          Phenotype Description
        </label>
        <textarea
          value={phenotypeText}
          onChange={(e) => setPhenotypeText(e.target.value)}
          className="w-full h-40 bg-white/5 border border-white/20 rounded-lg p-4 text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent resize-none"
          placeholder="Paste your microbiology description here..."
        />
        
        <div className="mt-4 flex items-center justify-between flex-wrap gap-4">
          <label className="flex items-center space-x-2 text-gray-300 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={useLLM}
              onChange={(e) => setUseLLM(e.target.checked)}
              className="w-4 h-4 rounded border-white/20 bg-white/5 text-blue-500 focus:ring-2 focus:ring-blue-400"
            />
            <span>Enable LLM Parser (Finetuned Flan-t5-base)</span>
          </label>
          
          <button
            onClick={analyzeText}
            disabled={loading}
            className="px-6 py-3 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-400/30 rounded-lg text-blue-200 font-medium transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                <span>Analyze & Identify</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 backdrop-blur-sm rounded-xl p-4 border border-red-500/30">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="text-red-300 font-medium mb-1">Error</h4>
              <p className="text-red-200 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {results && results.top5_table && results.top5_table.length > 0 && (
        <>
          {/* Top 5 Decision Table */}
          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <h3 className="text-xl font-semibold text-gray-200 mb-4 flex items-center space-x-2">
              <Database className="w-5 h-5" />
              <span>Top 5 Genus Predictions (Decision Table)</span>
            </h3>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/20">
                    <th className="text-left py-3 px-4 text-gray-300 font-medium">Genus</th>
                    <th className="text-left py-3 px-4 text-gray-300 font-medium">Probability % (Top 5)</th>
                    <th className="text-left py-3 px-4 text-gray-300 font-medium">Probability (Odds)</th>
                    <th className="text-left py-3 px-4 text-gray-300 font-medium">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {results.top5_table.map((row, idx) => (
                    <tr key={idx} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="py-3 px-4 text-gray-200 font-medium">{row[0]}</td>
                      <td className="py-3 px-4 text-gray-300">{row[1]}</td>
                      <td className="py-3 px-4 text-gray-300">{row[2]}</td>
                      <td className={`py-3 px-4 font-medium ${getConfidenceColor(row[3])}`}>
                        <span className="inline-flex items-center space-x-1">
                          <span>{getConfidenceEmoji(row[3])}</span>
                          <span>{row[3]}</span>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Genus Cards with Accordions */}
          <div className="space-y-3">
            <h3 className="text-xl font-semibold text-gray-200 mb-2">Detailed Results</h3>
            {results.genus_cards && results.genus_cards.map((card, idx) => (
              <div
                key={idx}
                className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/10 overflow-hidden hover:border-white/20 transition-all"
              >
                <button
                  onClick={() => setExpandedGenus(expandedGenus === idx ? -1 : idx)}
                  className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-white/5 transition-colors"
                >
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{getConfidenceEmoji(card.decision_band)}</span>
                      <div>
                        <h4 className="text-lg font-semibold text-gray-200">
                          Rank {idx + 1}: {card.genus}
                        </h4>
                        <p className="text-sm text-gray-400">
                          Combined: {card.combined_percent.toFixed(1)}% • {card.decision_band}
                        </p>
                      </div>
                    </div>
                  </div>
                  {expandedGenus === idx ? (
                    <ChevronUp className="w-5 h-5 text-gray-400 flex-shrink-0" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-gray-400 flex-shrink-0" />
                  )}
                </button>
                
                {expandedGenus === idx && (
                  <div className="px-6 pb-6 space-y-4 border-t border-white/10 pt-4">
                    {/* Combined Score Progress Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-gray-400">Combined Score</span>
                        <span className="text-gray-200 font-medium">{card.combined_percent.toFixed(1)}%</span>
                      </div>
                      <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${getProgressBarColor(card.combined_percent)}`}
                          style={{ width: `${Math.min(card.combined_percent, 100)}%` }}
                        />
                      </div>
                    </div>

                    {/* Tri-Fusion and ML Breakdown */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                        <span className="text-gray-400 text-sm block mb-1">Tri-Fusion (Blended)</span>
                        <p className="text-gray-200 font-semibold text-lg">{card.tri_fusion_blended_percent.toFixed(1)}%</p>
                      </div>
                      <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                        <span className="text-gray-400 text-sm block mb-1">ML Probability</span>
                        <p className="text-gray-200 font-semibold text-lg">{card.ml_prob_percent.toFixed(1)}%</p>
                      </div>
                    </div>

                    {/* Decision Confidence Badge */}
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-400 text-sm">Decision Confidence:</span>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(card.decision_band)} bg-white/5 border border-current`}>
                        {card.decision_band}
                      </span>
                    </div>

                    {/* RAG Interpretation */}
                    {card.rag_text && (
                      <div className="mt-4 p-4 bg-white/5 rounded-lg border border-white/10">
                        <h5 className="text-sm font-semibold text-gray-300 mb-3 flex items-center space-x-2">
                          <Info className="w-4 h-4" />
                          <span>RAG Interpretation (Genus-Level)</span>
                        </h5>
                        <div className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                          {card.rag_text}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );

  const renderTraining = () => (
    <div className="space-y-4">
      <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
        <p className="text-gray-300 mb-6">
          Evaluate parsers, train from gold tests, tune parser weights, train the genus-level model, 
          build the RAG index, and commit artifacts back to the HF Space repository.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <button
            onClick={() => runTraining('evaluate-parsers', 'Evaluate Parsers')}
            disabled={loading}
            className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/20 rounded-lg text-gray-200 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Beaker className="w-5 h-5" />
            <span>📊 Evaluate Parsers</span>
          </button>
          
          <button
            onClick={() => runTraining('train-gold', 'Train from Gold Tests')}
            disabled={loading}
            className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/20 rounded-lg text-gray-200 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <BookOpen className="w-5 h-5" />
            <span>🧬 Train from Gold Tests</span>
          </button>
          
          <button
            onClick={() => runTraining('train-weights', 'Train Parser Weights')}
            disabled={loading}
            className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/20 rounded-lg text-gray-200 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Database className="w-5 h-5" />
            <span>⚖️ Train Parser Weights</span>
          </button>
          
          <button
            onClick={() => runTraining('train-genus', 'Train Genus Model')}
            disabled={loading}
            className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/20 rounded-lg text-gray-200 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Database className="w-5 h-5" />
            <span>🧬 Train Genus Model</span>
          </button>
          
          <button
            onClick={() => runTraining('build-rag-index', 'Build RAG Index')}
            disabled={loading}
            className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/20 rounded-lg text-gray-200 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <BookOpen className="w-5 h-5" />
            <span>🧱 Build RAG Index</span>
          </button>
          
          <button
            onClick={() => runTraining('commit-hf', 'Commit to HuggingFace')}
            disabled={loading}
            className="px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/20 rounded-lg text-gray-200 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Upload className="w-5 h-5" />
            <span>⬆️ Commit to HF</span>
          </button>
        </div>
      </div>

      {trainingOutput && (
        <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
          <h3 className="text-lg font-semibold text-gray-200 mb-4 flex items-center space-x-2">
            {trainingOutput.error ? (
              <>
                <AlertCircle className="w-5 h-5 text-red-400" />
                <span>Training Output (Error)</span>
              </>
            ) : (
              <>
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span>Training Output (Success)</span>
              </>
            )}
          </h3>
          <pre className="bg-black/30 p-4 rounded-lg overflow-x-auto text-sm text-gray-300 max-h-96 overflow-y-auto">
            {JSON.stringify(trainingOutput, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );

  return (
    <div 
      className="min-h-screen p-4 md:p-6"
      style={{
        backgroundImage: 'url(/eph.jpeg)',
        backgroundSize: 'cover',
        backgroundPosition: 'center center',
        backgroundAttachment: 'fixed',
        position: 'relative'
      }}
    >
      {/* Overlay for glassmorphism effect - matches Gradio CSS */}
      <div 
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.55)',
          backdropFilter: 'blur(14px)',
          WebkitBackdropFilter: 'blur(14px)',
          zIndex: 0,
          pointerEvents: 'none'
        }}
      />
      
      <div className="max-w-7xl mx-auto relative z-10">
        <div className="bg-white/5 backdrop-blur-lg rounded-2xl border border-white/10 overflow-hidden shadow-2xl">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 p-6 md:p-8 border-b border-white/10">
            <h1 className="text-3xl md:text-4xl font-bold text-white mb-2 flex items-center space-x-3">
              <span className="text-4xl md:text-5xl">🧫</span>
              <span>BactAI-D</span>
            </h1>
            <p className="text-gray-300 text-base md:text-lg">
              Microbiology Phenotype Identification
            </p>
            <p className="text-gray-400 text-sm mt-2">
              Database updated: {dbLastUpdated || 'Loading...'}
            </p>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-white/10">
            <button
              onClick={() => setActiveTab('identify')}
              className={`flex-1 px-4 md:px-6 py-4 text-center transition-all text-sm md:text-base ${
                activeTab === 'identify'
                  ? 'bg-blue-500/20 text-blue-200 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:bg-white/5'
              }`}
            >
              🧬 Identification
            </button>
            <button
              onClick={() => setActiveTab('training')}
              className={`flex-1 px-4 md:px-6 py-4 text-center transition-all text-sm md:text-base ${
                activeTab === 'training'
                  ? 'bg-blue-500/20 text-blue-200 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:bg-white/5'
              }`}
            >
              📚 Training & Sync
            </button>
          </div>

          {/* Content */}
          <div className="p-4 md:p-8">
            {activeTab === 'identify' && renderIdentification()}
            {activeTab === 'training' && renderTraining()}
          </div>

          {/* Footer */}
          <div className="bg-white/5 border-t border-white/10 p-4 text-center">
            <p className="text-gray-400 text-sm">
              Built by <span className="font-semibold text-gray-300">Zain Asad</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BactAIApp;