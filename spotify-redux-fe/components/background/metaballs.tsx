import React from 'react';
import { Platform, StyleSheet, View } from 'react-native';
import {
  BlurMaskFilter,
  Canvas,
  Circle,
  Group,
  LinearGradient as SkiaLinearGradient,
  Rect,
  Skia,
  vec,
} from '@shopify/react-native-skia';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { useAnimatedStyle, useSharedValue, withRepeat, withTiming } from 'react-native-reanimated';

import { useColorScheme } from '@/hooks/use-color-scheme';

type BallSpec = {
  radius: number;
  base: { x: number; y: number };
  amp: { x: number; y: number };
  speed: number; // radians per second
  phase: number; // initial phase
};

function createSpecs(width: number, height: number): BallSpec[] {
  const min = Math.min(width, height);
  const r = (n: number) => min * (0.08 + (n % 5) * 0.02);
  return new Array(8).fill(0).map((_, i) => ({
    radius: r(i),
    base: { x: width * (0.2 + (i % 4) * 0.2), y: height * (0.18 + ((i + 1) % 3) * 0.18) },
    amp: { x: width * (0.06 + (i % 3) * 0.02), y: height * (0.05 + (i % 2) * 0.03) },
    speed: 0.3 + (i % 5) * 0.07,
    phase: i * 0.9,
  }));
}

type Props = {
  width?: number;
  height?: number;
};

export function MetaballsBackground({ width = 420, height = 360 }: Props) {
  // Skia isn't available on web/Expo Go or when native module is missing; bail out to avoid runtime errors
  const skiaReady = Skia?.PictureRecorder != null;
  if (Platform.OS === 'web' || !skiaReady) {
    return <FallbackGradient height={height} />;
  }

  return <MetaballsSkia width={width} height={height} />;
}

function MetaballsSkia({ width, height }: { width: number; height: number }) {
  const colorScheme = useColorScheme() ?? 'light';
  const isDark = colorScheme === 'dark';
  const specs = React.useMemo(() => createSpecs(width, height), [width, height]);
  const [time, setTime] = React.useState(0);

  // Drive time with rAF; simple state to stay compatible with older Skia builds
  React.useEffect(() => {
    let frame = 0;
    const loop = (ts: number) => {
      setTime(ts);
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, []);

  // Animated centers for each ball
  const centers = React.useMemo(() => {
    const t = time / 1000;
    return specs.map((s) => {
      const x = s.base.x + Math.sin(t * s.speed + s.phase) * s.amp.x;
      const y = s.base.y + Math.cos(t * (s.speed * 0.9) + s.phase) * s.amp.y;
      return vec(x, y);
    });
  }, [specs, time]);

  const bgTop = isDark ? '#0c0f0d' : '#f7faf8';
  const bgBottom = isDark ? '#0f1411' : '#eef7f2';
  const ink = isDark ? 'rgba(29,185,84,0.8)' : 'rgba(29,185,84,0.65)';
  const accent = isDark ? 'rgba(88,86,214,0.7)' : 'rgba(88,86,214,0.55)';

  return (
    <View
      pointerEvents="none"
      style={{ position: 'absolute', top: 0, left: 0, right: 0, height, zIndex: -1 }}
    >
      <Canvas style={{ width: '100%', height }}>
        {/* Background gradient */}
        <Rect x={0} y={0} width={width} height={height}>
          <SkiaLinearGradient start={vec(0, 0)} end={vec(width, height)} colors={[bgTop, bgBottom]} />
        </Rect>

        {/* Metaballs layer with blur and additive blending */}
        <Group
          layer={
            // Apply strong blur to fuse edges
            <BlurMaskFilter blur={40} style="normal" />
          }
          blendMode="plus"
          opacity={0.95}
        >
          {centers.map((c, i) => (
            <Circle key={i} c={c} r={specs[i].radius} color={i % 2 === 0 ? ink : accent} />
          ))}
        </Group>

        {/* Soft highlight wash */}
        <Rect x={0} y={0} width={width} height={height} opacity={isDark ? 0.25 : 0.2}>
          <SkiaLinearGradient
            start={vec(0, height * 0.15)}
            end={vec(width, height)}
            colors={[isDark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.45)', 'transparent']}
          />
        </Rect>
      </Canvas>
    </View>
  );
}

function FallbackGradient({ height }: { height: number }) {
  const scheme = useColorScheme() ?? 'light';
  const isDark = scheme === 'dark';
  const shift = useSharedValue(0);

  React.useEffect(() => {
    shift.value = withRepeat(withTiming(1, { duration: 12000 }), -1, true);
  }, [shift]);

  const animated = useAnimatedStyle(() => ({
    transform: [{ translateX: (shift.value - 0.5) * 60 }],
    opacity: isDark ? 0.3 : 0.35,
  }));

  return (
    <View
      pointerEvents="none"
      style={{ position: 'absolute', top: 0, left: 0, right: 0, height, zIndex: -1 }}
    >
      <LinearGradient
        colors={isDark ? ['#0c0f0d', '#0f1411'] : ['#f7faf8', '#eef7f2']}
        start={[0, 0]}
        end={[1, 1]}
        style={StyleSheet.absoluteFill}
      />
      <Animated.View style={[StyleSheet.absoluteFill, animated]}>
        <LinearGradient
          colors={isDark ? ['rgba(29,185,84,0.18)', 'rgba(88,86,214,0.18)', 'rgba(244,67,54,0.12)'] : ['rgba(29,185,84,0.16)', 'rgba(88,86,214,0.16)', 'rgba(244,67,54,0.1)']}
          start={[0.1, 0]}
          end={[1, 1]}
          style={StyleSheet.absoluteFill}
        />
      </Animated.View>
    </View>
  );
}


