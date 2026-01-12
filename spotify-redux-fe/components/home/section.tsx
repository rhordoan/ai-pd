import React from 'react';
import { Pressable, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';

type SectionProps = {
  title: string;
  onSeeAll?: () => void;
  children: React.ReactNode;
};

export function Section({ title, onSeeAll, children }: SectionProps) {
  return (
    <View style={styles.section}>
      <View style={styles.wrapper}>
        <View style={styles.headerRow}>
          <ThemedText type="subtitle" style={styles.title}>
            {title}
          </ThemedText>
          {onSeeAll && (
            <Pressable accessibilityRole="button" onPress={onSeeAll}>
              <ThemedText type="link">See all</ThemedText>
            </Pressable>
          )}
        </View>
        {children}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  section: {
    gap: 10,
  },
  wrapper: {
    gap: 12,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  title: {
    fontWeight: '900',
    letterSpacing: -0.1,
  },
});


