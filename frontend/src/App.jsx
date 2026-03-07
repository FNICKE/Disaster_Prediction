import React, { useState, useEffect } from 'react';
import { checkHealth, predictFlood } from './api';
import { 
    Activity, 
    AlertTriangle, 
    CheckCircle2, 
    CloudRain, 
    Droplets, 
    Info, 
    Loader2, 
    Settings,
    ThermometerSun,
    TrendingUp,
    Wind
} from 'lucide-react';

const TOP_FEATURES_META = {
    CoastalVulnerability: { icon: Wind, label: "Coastal Vulnerability", desc: "Susceptibility to sea/storm surges" },
    DamsQuality: { icon: Settings, label: "Dams Quality", desc: "Structural integrity of dams" },
    DeterioratingInfrastructure: { icon: AlertTriangle, label: "Deteriorating Infra", desc: "Aging roads, bridges, pipes" },
    Watersheds: { icon: Droplets, label: "Watersheds", desc: "Health of local catchments" },
    Landslides: { icon: TrendingUp, label: "Landslides", desc: "Risk of land movement" }
};

// All 20 features required by ML backend.
const ALL_FEATURE_KEYS = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement",
    "Deforestation", "Urbanization", "ClimateChange", "DamsQuality",
    "Siltation", "AgriculturalPractices", "Encroachments",
    "IneffectiveDisasterPreparedness", "DrainageSystems",
    "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
];

// Initial state - setting baseline config for the Top 5 inputs
const INITIAL_FEATURES = Object.keys(TOP_FEATURES_META).reduce((acc, key) => {
    acc[key] = 5; 
    return acc;
}, {});

function App() {
    const [features, setFeatures] = useState(INITIAL_FEATURES);
    const [status, setStatus] = useState({ state: 'checking', msg: 'Connecting to backend...' });
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Initial Health Check
    useEffect(() => {
        const checkConnection = async () => {
            const health = await checkHealth();
            if (health.model_ready) {
                setStatus({ state: 'online', msg: 'Connected to ML Prediction Engine' });
                // Make initial prediction with defaults
                handlePredict(INITIAL_FEATURES);
            } else if (health.status === 'offline') {
                setStatus({ state: 'offline', msg: 'Cannot reach backend. Is Flask running?' });
                setError("Network error: Could not connect to API at localhost:5000");
            } else {
                setStatus({ state: 'error', msg: health.message || 'Model not loaded' });
                setError(health.message || "Model not loaded on backend.");
            }
        };
        checkConnection();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleSliderChange = (key, value) => {
        setFeatures(prev => ({
            ...prev,
            [key]: parseInt(value)
        }));
    };

    const handlePredict = async (featuresToUse = features) => {
        setLoading(true);
        setError(null);
        
        // ML Model requires exactly 20 features.
        // We auto-fill non-UI features with the dataset average (5).
        const fullPayload = ALL_FEATURE_KEYS.reduce((acc, key) => {
            acc[key] = featuresToUse[key] !== undefined ? featuresToUse[key] : 5;
            return acc;
        }, {});

        try {
            const res = await predictFlood(fullPayload);
            setPrediction(res);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const getRiskColor = (level) => {
        switch(level) {
            case 'Low': return 'text-emerald-500 bg-emerald-50 border-emerald-200';
            case 'Moderate': return 'text-yellow-500 bg-yellow-50 border-yellow-200';
            case 'High': return 'text-orange-500 bg-orange-50 border-orange-200';
            case 'Very High': return 'text-red-500 bg-red-50 border-red-200';
            default: return 'text-gray-500 bg-gray-50 border-gray-200';
        }
    };

    const getRiskGradient = (level) => {
        switch(level) {
            case 'Low': return 'from-emerald-400 to-emerald-600';
            case 'Moderate': return 'from-yellow-400 to-yellow-600';
            case 'High': return 'from-orange-400 to-orange-600';
            case 'Very High': return 'from-red-500 to-pink-600';
            default: return 'from-blue-500 to-indigo-600';
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 font-sans selection:bg-blue-100">
            {/* Header */}
            <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                            <CloudRain className="w-5 h-5 text-white" />
                        </div>
                        <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-900 to-indigo-800">
                            FloodRisk AI
                        </h1>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                        {status.state === 'online' && <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                        {status.state === 'checking' && <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />}
                        {(status.state === 'offline' || status.state === 'error') && <AlertTriangle className="w-4 h-4 text-red-500" />}
                        <span className={`
                            ${status.state === 'online' ? 'text-emerald-700' : ''}
                            ${status.state === 'checking' ? 'text-blue-700' : ''}
                            ${(status.state === 'offline' || status.state === 'error') ? 'text-red-700' : ''}
                        `}>
                            {status.msg}
                        </span>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 h-[calc(100vh-4rem)] flex flex-col items-center justify-center">
                <div className="grid lg:grid-cols-12 gap-8 items-start w-full">
                    
                    {/* Left Column: Input Form (Scrollable) */}
                    <div className="lg:col-span-8 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col h-[85vh]">
                        <div className="p-6 border-b border-slate-100 bg-white z-10">
                            <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                                <Settings className="w-5 h-5 text-indigo-500" />
                                Feature Configuration
                            </h2>
                            <p className="text-sm text-slate-500 mt-1">Adjust environmental, structural, and political factors to see real-time ML risk predictions.</p>
                        </div>
                        
                        <div className="p-6 overflow-y-auto flex-1 bg-slate-50/50">
                            <div className="grid sm:grid-cols-2 gap-x-8 gap-y-6">
                                {Object.entries(TOP_FEATURES_META).map(([key, meta]) => {
                                    const Icon = meta.icon;
                                    return (
                                        <div key={key} className="group relative bg-white p-4 rounded-xl border border-slate-200 hover:border-indigo-300 transition-colors shadow-sm hover:shadow-md">
                                            <div className="flex justify-between items-start mb-2">
                                                <div className="flex items-center gap-2">
                                                    <div className="p-1.5 rounded-md bg-indigo-50 text-indigo-600 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                                                        <Icon className="w-4 h-4" />
                                                    </div>
                                                    <label className="text-sm font-medium text-slate-700">{meta.label}</label>
                                                </div>
                                                <span className="text-sm font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-full">
                                                    {features[key]}
                                                </span>
                                            </div>
                                            <p className="text-xs text-slate-500 mb-4 h-4">{meta.desc}</p>
                                            
                                            <div className="relative pt-1">
                                                <input 
                                                    type="range" 
                                                    min="0" 
                                                    max="16" 
                                                    value={features[key]} 
                                                    onChange={(e) => handleSliderChange(key, e.target.value)}
                                                    className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        <div className="p-6 border-t border-slate-100 bg-white">
                            <button 
                                onClick={() => handlePredict()}
                                disabled={loading || status.state !== 'online'}
                                className="w-full sm:w-auto relative flex items-center justify-center px-8 py-3.5 border border-transparent text-sm font-semibold rounded-xl text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-indigo-500/30 overflow-hidden group"
                            >
                                <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
                                <span className="relative flex items-center gap-2">
                                    {loading ? (
                                        <>
                                            <Loader2 className="w-5 h-5 animate-spin"/>
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <Activity className="w-5 h-5"/>
                                            Calculate Risk Priority
                                        </>
                                    )}
                                </span>
                            </button>
                            {error && (
                                <div className="mt-4 flex items-center gap-2 text-sm text-red-600 bg-red-50 p-3 rounded-lg border border-red-100">
                                    <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                                    {error}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Right Column: Prediction Results sticky */}
                    <div className="lg:col-span-4 sticky top-24">
                        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden relative">
                            {/* Decorative background */}
                            <div className={`absolute inset-0 opacity-10 bg-gradient-to-br ${prediction ? getRiskGradient(prediction.risk_level) : 'from-slate-400 to-slate-200'}`}></div>
                            
                            <div className="p-6 relative">
                                <h2 className="text-lg font-semibold text-slate-800 mb-6 flex items-center gap-2">
                                    <Activity className="w-5 h-5 text-slate-400" />
                                    Prediction Results
                                </h2>

                                {!prediction && !loading && !error && (
                                    <div className="flex flex-col items-center justify-center py-12 text-center">
                                        <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                                            <Info className="w-8 h-8 text-slate-300" />
                                        </div>
                                        <p className="text-slate-500 font-medium">No calculation yet</p>
                                        <p className="text-sm text-slate-400 mt-1">Adjust features and click Calculate</p>
                                    </div>
                                )}

                                {loading && (
                                    <div className="flex flex-col items-center justify-center py-12 text-center">
                                        <Loader2 className="w-12 h-12 text-indigo-500 animate-spin mb-4" />
                                        <p className="text-indigo-600 font-medium animate-pulse">Running XGBoost Model...</p>
                                    </div>
                                )}

                                {prediction && !loading && (
                                    <div className="space-y-6">
                                        {/* Main dial / gauge */}
                                        <div className="relative flex justify-center">
                                            <svg className="w-48 h-48 transform -rotate-90">
                                                <circle 
                                                    cx="96" cy="96" r="88" 
                                                    stroke="currentColor" 
                                                    strokeWidth="12" 
                                                    fill="transparent" 
                                                    className="text-slate-100"
                                                />
                                                <circle 
                                                    cx="96" cy="96" r="88" 
                                                    stroke="currentColor" 
                                                    strokeWidth="12" 
                                                    fill="transparent" 
                                                    strokeDasharray={2 * Math.PI * 88}
                                                    strokeDashoffset={2 * Math.PI * 88 * (1 - prediction.probability)}
                                                    className={`transition-all duration-1000 ease-out ${getRiskColor(prediction.risk_level).split(' ')[0]}`}
                                                    strokeLinecap="round"
                                                />
                                            </svg>
                                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                                <span className="text-4xl font-black text-slate-800 tracking-tighter">
                                                    {(prediction.probability * 100).toFixed(1)}<span className="text-xl text-slate-400">%</span>
                                                </span>
                                                <span className="text-xs uppercase tracking-widest text-slate-400 font-semibold mt-1">Probability</span>
                                            </div>
                                        </div>

                                        <div className={`p-5 rounded-xl border flex flex-col items-center justify-center text-center ${getRiskColor(prediction.risk_level)} transform transition-all duration-500`}>
                                            <span className="text-sm uppercase tracking-wider font-bold opacity-80 mb-1">Risk Level</span>
                                            <span className="text-3xl font-black">{prediction.risk_level}</span>
                                        </div>
                                        
                                        <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                                            <h3 className="text-xs uppercase font-bold text-slate-400 tracking-wider mb-3">Model Analysis</h3>
                                            <div className="space-y-3">
                                                <div className="flex justify-between items-center text-sm">
                                                    <span className="text-slate-500">Classification</span>
                                                    <span className="font-semibold text-slate-700">{prediction.label}</span>
                                                </div>
                                                <div className="flex justify-between items-center text-sm">
                                                    <span className="text-slate-500">Engine</span>
                                                    <span className="font-mono text-xs bg-slate-200 text-slate-600 px-2 py-0.5 rounded">XGBClassifier</span>
                                                </div>
                                                <div className="flex justify-between items-center text-sm">
                                                    <span className="text-slate-500">Confidence</span>
                                                    <span className="font-semibold text-emerald-600">High (CV AUC 1.00)</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                        
                        {!prediction && (
                            <div className="mt-6 bg-blue-50/50 rounded-xl p-5 border border-blue-100">
                                <p className="text-sm text-blue-800 leading-relaxed">
                                    <span className="font-semibold">Tip:</span> The model detects compounding risks. For example, high <b>Monsoon Intensity</b> paired with poor <b>Topography Drainage</b> will drastically increase the probability of severe flooding.
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;
