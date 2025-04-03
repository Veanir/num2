"use client";

import React, { useRef, useState } from 'react';
import CanvasDraw from 'react-canvas-draw';

export default function Home() {
  const canvasRef = useRef<CanvasDraw>(null);
  const [predictedEmoji, setPredictedEmoji] = useState<string>("❓");

  // Helper function to convert data URL to Blob
  const dataURLtoBlob = (dataurl: string): Blob | null => {
    const arr = dataurl.split(',');
    if (arr.length < 2) return null;
    const mimeMatch = arr[0].match(/:(.*?);/);
    if (!mimeMatch) return null;
    const mime = mimeMatch[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  }

  // --- Predict Handler ---
  const handlePredict = async () => {
    if (!canvasRef.current) return;
    setPredictedEmoji("⏳");

    try {
      // Cast to any specifically for this method call
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const originalImageDataUrl = (canvasRef.current as any).getDataURL("image/png", false, "#FFFFFF");

      const imageBlob = dataURLtoBlob(originalImageDataUrl);
      if (!imageBlob) {
        throw new Error("Failed to convert canvas data to Blob.");
      }

      // Create FormData and append the image blob
      const formData = new FormData();
      formData.append('image_input', imageBlob, 'drawing.png');

      // Send multipart/form-data request to the NEW BentoCloud URL
      const response = await fetch('https://emoji-classifier-service-7sra-566d3747.mt-guc1.bentoml.ai/predict', {
        method: 'POST',
        // No 'Content-Type' header needed, fetch sets it for FormData
        body: formData // Send FormData object
      });

      if (!response.ok) {
        // Try to parse error response (might be JSON or plain text)
        let errorDetail = 'Unknown API error';
        try {
          const errorData = await response.json();
          errorDetail = errorData.error || JSON.stringify(errorData);
        } catch {
          errorDetail = await response.text(); // Fallback to text
        }
        console.error("API Error Response:", response.status, response.statusText, errorDetail);
        throw new Error(`API error: ${response.status} ${response.statusText} - ${errorDetail}`);
      }

      const result = await response.json(); // Expect JSON response on success
      if (result.error) {
        console.error("Prediction Error:", result.error);
        setPredictedEmoji("❌");
      } else {
        // Use the predicted_emoji field directly from the backend response
        setPredictedEmoji(result.predicted_emoji || "❓");
      }
      console.log("Prediction result:", result);

    } catch (error) {
      console.error("Prediction failed:", error);
      setPredictedEmoji("❌");
    }
  };

  // --- Clear Handler ---
  const handleClear = () => {
    canvasRef.current?.clear();
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12 md:p-24">
      <div className="flex flex-col items-center w-full">
        <h1 className="text-3xl md:text-4xl font-bold mb-6 md:mb-8">Draw an Emoji!</h1>

        <div className="mb-4 border-2 border-gray-400 rounded-lg overflow-hidden bg-white">
          <CanvasDraw
            ref={canvasRef}
            brushRadius={4}
            brushColor="#000000"
            lazyRadius={0}
            canvasWidth={300}
            canvasHeight={300}
            hideGrid={false}
            style={{ touchAction: 'none' }}
          />
        </div>

        <div className="flex gap-4 mb-4">
          <button
            onClick={handleClear}
            className="px-4 py-2 bg-gray-300 text-black rounded hover:bg-gray-400 transition-colors"
          >
            Clear
          </button>
          <button
            onClick={handlePredict}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          >
            Predict
          </button>
        </div>

        <div className="mt-4 text-2xl md:text-3xl bg-white/70 backdrop-blur-sm p-2 rounded">
          Predicted Emoji: <span id="result" className="font-mono ml-2">{predictedEmoji}</span>
        </div>
      </div>
    </main>
  );
}
