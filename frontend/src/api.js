/**
 * API service to connect to the Flask Backend
 */

const BASE_URL = 'http://127.0.0.1:5000';

export const checkHealth = async () => {
    try {
        const response = await fetch(`${BASE_URL}/health`);
        return await response.json();
    } catch (error) {
        console.error("Backend health check failed:", error);
        return { status: "offline", model_ready: false };
    }
}

export const getFeatures = async () => {
    try {
        const response = await fetch(`${BASE_URL}/features`);
        return await response.json();
    } catch (error) {
        console.error("Failed to fetch features:", error);
        return { features: [] };
    }
}

/**
 * Sends a prediction request using the named data format.
 * @param {Object} data - Dictionary of the 20 features
 */
export const predictFlood = async (data) => {
    try {
        const response = await fetch(`${BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ data })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Failed to predict');
        }
        
        return result;
    } catch (error) {
        console.error("Prediction failed:", error);
        throw error;
    }
}
