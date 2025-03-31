"use client"; // Add this directive for client-side interactivity

import React, { useRef, useState, useEffect, useCallback } from 'react';
import CanvasDraw from 'react-canvas-draw';
import Matter from 'matter-js';

export default function Home() {
  const canvasRef = useRef<CanvasDraw>(null);
  const [predictedEmoji, setPredictedEmoji] = useState<string>("❓");

  // --- Physics Engine State ---
  const matterContainerRef = useRef<HTMLDivElement>(null);
  const emojiCanvasRef = useRef<HTMLCanvasElement>(null); // Ref for the dedicated emoji canvas
  const engineRef = useRef<Matter.Engine | null>(null);
  const renderRef = useRef<Matter.Render | null>(null); // Keep Matter's renderer for physics (but keep it transparent)
  const runnerRef = useRef<Matter.Runner | null>(null);
  const [emojiBodies, setEmojiBodies] = useState<Matter.Body[]>([]); // Keep track of bodies

  // --- WebSocket State ---
  const ws = useRef<WebSocket | null>(null);

  // --- Function to add new emoji body ---
  // Define addEmojiToWorld BEFORE the main useEffect that depends on it
  const addEmojiToWorld = useCallback((emoji: string) => {
    console.log('>>> addEmojiToWorld called with:', emoji);
    if (!engineRef.current || !renderRef.current?.options.width) {
      console.log('>>> addEmojiToWorld: Engine or Render not ready');
      return;
    }

    const radius = 20; // Size of emoji circle
    const x = Math.random() * (renderRef.current.options.width - radius * 2) + radius;
    const y = -radius; // Start above the screen

    const body = Matter.Bodies.circle(x, y, radius, {
      restitution: 0.6, // Bounciness
      friction: 0.1,
      render: {
        // Use sprite with text for emoji rendering
        sprite: {
          texture: '', // We'll draw text instead
          xScale: 1,
          yScale: 1
        },
        // Custom rendering logic to draw the emoji text
        fillStyle: 'transparent' // Make circle transparent
      },
      // Store emoji on the body itself for custom rendering
      plugin: { emoji: emoji } 
    });

    Matter.World.add(engineRef.current.world, body);
    setEmojiBodies(prev => [...prev, body]);

    // Optional: Remove old emojis to prevent too many bodies
    // This is a simple example, might need refinement
    if (emojiBodies.length > 50) {
       const oldestBody = emojiBodies[0];
       Matter.World.remove(engineRef.current.world, oldestBody);
       setEmojiBodies(prev => prev.slice(1));
    }

  }, [emojiBodies]); // Depend on emojiBodies to manage length

  // --- Initialize Physics Engine and WebSocket ---
  useEffect(() => {
    if (typeof window === 'undefined' || !matterContainerRef.current || !emojiCanvasRef.current) return; // Wait for both refs

    console.log("Setting up Physics and WebSocket...");

    const containerElement = matterContainerRef.current;
    const emojiCanvas = emojiCanvasRef.current;

    // Ensure canvas matches container size
    emojiCanvas.width = containerElement.clientWidth;
    emojiCanvas.height = containerElement.clientHeight;

    // --- Physics Setup ---
    const engine = Matter.Engine.create({ gravity: { y: 0.4 } });
    engineRef.current = engine;
    
    // Configure Matter's renderer (transparent, no wireframes)
    const render = Matter.Render.create({
      element: containerElement, // Render into the container DIV
      canvas: containerElement.querySelector('canvas') || undefined, // Try to reuse canvas if one exists from previous run
      engine: engine,
      options: { 
        width: containerElement.clientWidth,
        height: containerElement.clientHeight,
        wireframes: false, 
        background: 'transparent', // Keep Matter's canvas transparent
        // Disable Matter's own rendering loop since we draw separately?
        // No, keep it running for physics, just make it invisible
      }
    });
    renderRef.current = render;

    // Create ground and walls (boundaries)
    const ground = Matter.Bodies.rectangle(
      render.options.width! / 2, 
      render.options.height! + 25, // Position below viewport
      render.options.width! + 50,
      50, 
      { isStatic: true, render: { visible: false } }
    );
    const leftWall = Matter.Bodies.rectangle(
      -25, 
      render.options.height! / 2, 
      50, 
      render.options.height! + 50, 
      { isStatic: true, render: { visible: false } }
    );
    const rightWall = Matter.Bodies.rectangle(
      render.options.width! + 25, 
      render.options.height! / 2, 
      50, 
      render.options.height! + 50, 
      { isStatic: true, render: { visible: false } }
    );
    Matter.World.add(engine.world, [ground, leftWall, rightWall]);

    // Start the renderer
    Matter.Render.run(render);

    // Create runner
    const runner = Matter.Runner.create();
    runnerRef.current = runner;
    Matter.Runner.run(runner, engine);

    // --- WebSocket Connection ---
    // Create WebSocket connection instance using environment variable
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws'; // Fallback for local dev
    console.log("Connecting WebSocket to:", wsUrl);
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket Connected', socket);
    };
    socket.onclose = () => {
      console.log('WebSocket Disconnected', socket);
    };
    socket.onerror = (error) => {
      console.error('WebSocket Error:', error, socket);
    };
    socket.onmessage = (event) => {
      try {
        const emoji = event.data;
        console.log('>>> WebSocket Message Received:', emoji);
        addEmojiToWorld(emoji); 
      } catch (error) {
        console.error('Failed to process WebSocket message:', error);
      }
    };

    ws.current = socket;

    // --- Cleanup function (adjusted) ---
    return () => {
      console.log("Cleaning up Matter.js and WebSocket...");
      // Stop physics engine
      if (renderRef.current) {
        Matter.Render.stop(renderRef.current); // Stop the renderer
        // No need to remove renderRef.current.canvas as it might be reused or part of the div
      }
       if (engineRef.current) { // Clear world and engine
           Matter.World.clear(engineRef.current.world, false);
           Matter.Engine.clear(engineRef.current);
       }
      if (runnerRef.current) {
        Matter.Runner.stop(runnerRef.current);
      }
      // Close WebSocket
      console.log("Closing WebSocket:", ws.current);
      ws.current?.close(); 
      // Clear state and refs
      setEmojiBodies([]); 
      engineRef.current = null;
      renderRef.current = null;
      runnerRef.current = null;
      ws.current = null;

      // Clear the dedicated emoji canvas
      const ctx = emojiCanvasRef.current?.getContext('2d');
      if (ctx && emojiCanvasRef.current) {
        ctx.clearRect(0, 0, emojiCanvasRef.current.width, emojiCanvasRef.current.height);
      }
    };
  }, []); 

  // --- Custom Emoji Rendering on dedicated canvas ---
  useEffect(() => {
    // Don't run if refs aren't ready
    if (!engineRef.current || !emojiCanvasRef.current) return;
    
    const emojiCanvas = emojiCanvasRef.current;
    const ctx = emojiCanvas.getContext('2d');
    if (!ctx) {
      console.error("Emoji canvas context not found!");
      return;
    }

    let frameId: number; // To store animation frame ID

    // Custom render loop using requestAnimationFrame
    const renderLoop = () => {
      // Clear the dedicated emoji canvas each frame
      ctx.clearRect(0, 0, emojiCanvas.width, emojiCanvas.height);

      // Set drawing properties
      ctx.font = '32px Arial'; 
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Draw current bodies from state
      emojiBodies.forEach(body => {
        if (body.plugin.emoji) {
           // Use positions directly from the physics engine bodies
           ctx.fillText(body.plugin.emoji, body.position.x, body.position.y);
           // console.log(`>>> Drawing ${body.plugin.emoji} at ${body.position.x.toFixed(1)}, ${body.position.y.toFixed(1)}`);
        }
      });
      
      // Request next frame
      frameId = requestAnimationFrame(renderLoop);
    };

    // Start the loop
    renderLoop();

    // Cleanup function to stop the loop
    return () => {
      console.log("Stopping custom render loop");
      cancelAnimationFrame(frameId);
    };

  }, [emojiBodies]); // Re-run this effect if emojiBodies array changes (to draw the new set)

  // --- Predict Handler ---
  const handlePredict = async () => {
    if (!canvasRef.current) return;
    setPredictedEmoji("⏳"); // Set loading state immediately

    try {
      // 1. Get the drawing as data URL (PNG) with white background
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const originalImageDataUrl = (canvasRef.current as any).getDataURL("image/png", false, "#FFFFFF");
      
      // 2. Send ORIGINAL image data to the backend API (backend handles resizing/preprocessing)
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'; // Fallback for local dev
      console.log("Sending prediction to:", `${apiUrl}/predict`);
      const response = await fetch(`${apiUrl}/predict`, { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_data_url: originalImageDataUrl }) 
      });

      if (!response.ok) {
        // Handle API errors (e.g., model not loaded, processing error)
        const errorData = await response.json().catch(() => ({ detail: 'Unknown API error' }));
        console.error("API Error:", errorData);
        throw new Error(`API error: ${response.status} ${response.statusText} - ${errorData.detail || ''}`);
      }

      // 3. Parse the response and update the state (for the predicting user)
      const result = await response.json(); 
      setPredictedEmoji(result.predicted_emoji || "❓"); // Update with predicted emoji
      // The backend now also broadcasts via WebSocket, triggering addEmojiToWorld for everyone
      console.log("Prediction result (sent via HTTP):", result); 

    } catch (error) {
      console.error("Prediction failed:", error);
      setPredictedEmoji("❌"); // Show an error state
    }
  };

  // --- Clear Handler ---
  const handleClear = () => {
    canvasRef.current?.clear();
  };

  return (
    <main className="relative flex min-h-screen flex-col items-center justify-start pt-10 pb-24 px-4 md:px-12">
      {/* Container for Matter.js rendering (kept transparent) */}
      <div 
        ref={matterContainerRef} 
        className="absolute inset-0 z-0 overflow-hidden"
        style={{ width: '100%', height: '100%' }}
      />
      {/* Dedicated Canvas for Emojis - Overlaying Matter container */}
      <canvas 
        ref={emojiCanvasRef} 
        className="absolute inset-0 z-10 pointer-events-none" // z-10, above matter(z-0), below UI(z-20)
        style={{ width: '100%', height: '100%' }}
      />

      {/* UI Content - positioned above the canvases */}
      <div className="relative z-20 flex flex-col items-center w-full">
        <h1 className="text-3xl md:text-4xl font-bold mb-6 md:mb-8">Draw an Emoji!</h1>
        
        <div className="mb-4 border-2 border-gray-400 rounded-lg overflow-hidden bg-white">
          {/* Drawing Canvas */}
          <CanvasDraw
            ref={canvasRef}
            brushRadius={4}
            brushColor="#000000"
            lazyRadius={0}
            canvasWidth={300} 
            canvasHeight={300}
            hideGrid={false}
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
