import { Image } from 'expo-image';
import React, { useEffect } from 'react';
import { Pressable, StyleSheet, View, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, { useAnimatedStyle, useSharedValue, withDelay, withRepeat, withSpring, withTiming } from 'react-native-reanimated';

import { ThemedText } from '@/components/themed-text';
import { useColorScheme } from '@/hooks/use-color-scheme';

type PlaylistCardVariant = 'tile' | 'card';

type PlaylistCardProps = {
  variant?: PlaylistCardVariant;
  imageUri: string;
  title: string;
  subtitle?: string;
  onPress?: () => void;
  index?: number;
};

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export function PlaylistCard({
  variant = 'card',
  imageUri,
  title,
  subtitle,
  onPress,
  index = 0,
}: PlaylistCardProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const pressed = useSharedValue(0);
  const appear = useSharedValue(0);
  const shine = useSharedValue(0);

  useEffect(() => {
    // Stagger entrance for pleasant feel
    const delay = Math.min(index, 8) * 60;
    appear.value = 0;
    appear.value = withDelay(delay, withTiming(1, { duration: 420 }));
    // Loop shine for portfolio-like polish
    shine.value = 0;
    shine.value = withDelay(delay + 500, withRepeat(withTiming(1, { duration: 3600 }), -1, false));
  }, [appear, index, shine]);

  const animatedStyle = useAnimatedStyle(() => {
    const scaleTarget = pressed.value ? 0.96 : 1;
    const translateY = (variant === 'tile' ? 6 : 12) * (1 - appear.value);
    const opacity = appear.value;
    return {
      transform: [
        { scale: withSpring(scaleTarget, { stiffness: 400, damping: 28 }) },
        { translateY },
        { rotateZ: withSpring(pressed.value ? '-0.6deg' : '0deg', { stiffness: 320, damping: 26 }) },
      ],
      opacity,
    };
  });

  const isDark = colorScheme === 'dark';
  const surface = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)';

  if (variant === 'tile') {
    return (
      <AnimatedPressable
        accessibilityRole="button"
        accessibilityLabel={title}
        onPress={onPress}
        onPressIn={() => (pressed.value = 1)}
        onPressOut={() => (pressed.value = 0)}
        style={[styles.tile, { backgroundColor: surface }, animatedStyle]}
      >
        <Image
          source={{ uri: imageUri }}
          style={styles.tileImage}
          contentFit="cover"
          transition={200}
        />
        <ThemedText numberOfLines={1} style={styles.tileText}>
          {title}
        </ThemedText>
      </AnimatedPressable>
    );
  }

  return (
    <AnimatedPressable
      accessibilityRole="button"
      accessibilityLabel={title}
      onPress={onPress}
      onPressIn={() => {
        pressed.value = 1;
        if (Platform.OS === 'ios') void Haptics.selectionAsync();
      }}
      onPressOut={() => (pressed.value = 0)}
      style={[
        styles.card,
        { backgroundColor: surface },
        Platform.OS === 'web' ? styles.shadowWeb : styles.shadowNative,
        animatedStyle,
      ]}
    >
      <View style={styles.cardImageWrapper}>
        <Image source={{ uri: imageUri }} style={styles.cardImage} contentFit="cover" transition={200} />
        {/* Shine sweep */}
        <Animated.View
          style={[
            styles.shine,
            {
              pointerEvents: 'none',
              transform: [
                {
                  translateX: shine.value * 340 - 180, // sweep across
                },
                { rotateZ: '24deg' },
              ],
              opacity: 0.16,
            },
          ]}
        />
      </View>
      <View style={styles.cardTextWrapper}>
        <ThemedText numberOfLines={1} style={styles.cardTitle}>
          {title}
        </ThemedText>
        {!!subtitle && (
          <ThemedText numberOfLines={1} style={styles.cardSubtitle}>
            {subtitle}
          </ThemedText>
        )}
      </View>
    </AnimatedPressable>
  );
}

const styles = StyleSheet.create({
  // Horizontal tile used in "Recently played"
  tile: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 10,
    overflow: 'hidden',
    height: 64,
  },
  tileImage: {
    width: 64,
    height: 64,
  },
  tileText: {
    flex: 1,
    marginHorizontal: 12,
    fontWeight: '600',
  },

  // Vertical card used in carousels
  card: {
    width: 148,
    borderRadius: 14,
    padding: 10,
  },
  shadowNative: {
    // iOS shadow
    shadowColor: '#000',
    shadowOpacity: 0.18,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 6 },
    // Android
    elevation: 4,
  },
  shadowWeb: {
    boxShadow: '0px 6px 16px rgba(0,0,0,0.18)',
  },
  cardImageWrapper: {
    borderRadius: 10,
    overflow: 'hidden',
  },
  cardImage: {
    width: '100%',
    height: 140,
    borderRadius: 10,
  },
  shine: {
    position: 'absolute',
    top: -40,
    left: -80,
    width: 120,
    height: 260,
    borderRadius: 20,
    backgroundColor: '#ffffff',
  },
  cardTextWrapper: {
    marginTop: 10,
    gap: 2,
  },
  cardTitle: {
    fontWeight: '600',
  },
  cardSubtitle: {
    opacity: 0.7,
    fontSize: 13,
  },
});


