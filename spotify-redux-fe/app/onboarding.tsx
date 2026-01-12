import React, { useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Image } from 'expo-image';

import { ThemedText } from '@/components/themed-text';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { useAuth } from '@/providers/auth';
import {
  getColdStartRecommendations,
  getLikes,
  getSeeds,
  searchTracks,
  toggleLike,
  type RecommendationItem,
  type SearchResult,
} from '@/lib/api';

const MIN_SEEDS = 3;
const MAX_SEEDS = 5;
const FALLBACK_COVER = 'https://via.placeholder.com/120x120.png?text=Track';

type Step = 'pick' | 'results';

export default function OnboardingScreen() {
  const router = useRouter();
  const isDark = (useColorScheme() ?? 'light') === 'dark';
  const { token, username, logout, ready: authReady } = useAuth();

  const [step, setStep] = useState<Step>('pick');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [seeds, setSeeds] = useState<string[]>([]);

  const [recs, setRecs] = useState<RecommendationItem[]>([]);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);

  const [likes, setLikes] = useState<Set<string>>(new Set());

  const palette = useMemo(() => {
    const bg: readonly [string, string, string] = isDark
      ? ['#050606', '#090f0c', '#050606']
      : ['#050606', '#090f0c', '#050606'];
    const glass = 'rgba(255,255,255,0.06)';
    const stroke = 'rgba(255,255,255,0.12)';
    const text = '#ffffff';
    const muted = 'rgba(255,255,255,0.72)';
    const muted2 = 'rgba(255,255,255,0.55)';
    return { bg, glass, stroke, text, muted, muted2 };
  }, [isDark]);

  // One-time guard: if already seeded, skip this screen.
  useEffect(() => {
    if (!authReady) return;
    if (!token) {
      router.replace('/login');
      return;
    }
    getSeeds(token)
      .then((res) => {
        if (res.seeds && res.seeds.length > 0) {
          router.replace('/(tabs)');
        }
      })
      .catch(() => {
        // ignore; allow onboarding
      });

    getLikes(token)
      .then((res) => setLikes(new Set(res.likes)))
      .catch(() => setLikes(new Set()));
  }, [authReady, token, router]);

  // Debounced catalog-only search
  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      setSearchError(null);
      return;
    }
    setSearchLoading(true);
    setSearchError(null);
    const handle = setTimeout(() => {
      searchTracks(query, 12)
        .then((r) => setResults(r))
        .catch((e: Error) => setSearchError(e.message))
        .finally(() => setSearchLoading(false));
    }, 220);
    return () => clearTimeout(handle);
  }, [query]);

  const addSeed = (songId: string) => {
    setSeeds((prev) => {
      if (prev.includes(songId) || prev.length >= MAX_SEEDS) return prev;
      return [...prev, songId];
    });
  };

  const removeSeed = (songId: string) => setSeeds((prev) => prev.filter((s) => s !== songId));

  const isLiked = (songId: string | undefined) => (songId ? likes.has(songId) : false);

  const toggleLikeSong = async (songId: string) => {
    if (!token) return;
    setLikes((prev) => {
      const next = new Set(prev);
      if (next.has(songId)) next.delete(songId);
      else next.add(songId);
      return next;
    });
    try {
      const res = await toggleLike(token, songId);
      setLikes((prev) => {
        const next = new Set(prev);
        if (res.liked) next.add(songId);
        else next.delete(songId);
        return next;
      });
    } catch {
      // revert on failure
      setLikes((prev) => {
        const next = new Set(prev);
        if (next.has(songId)) next.delete(songId);
        else next.add(songId);
        return next;
      });
    }
  };

  const submit = async () => {
    if (!token) return;
    setRecLoading(true);
    setRecError(null);
    try {
      const res = await getColdStartRecommendations(seeds, 12, token);
      setRecs(res.results);
      setStep('results');
      // clear selection for a clean next step UI
      setSeeds([]);
    } catch (e) {
      setRecError((e as Error).message);
    } finally {
      setRecLoading(false);
    }
  };

  const finish = () => router.replace('/(tabs)');

  return (
    <View style={styles.page}>
      <LinearGradient colors={palette.bg} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={StyleSheet.absoluteFill} />
      <View style={styles.bgGlowWrap} pointerEvents="none">
        <LinearGradient
          colors={['rgba(29,185,84,0.22)', 'rgba(88,86,214,0.16)', 'rgba(255,255,255,0.00)']}
          start={{ x: 0.1, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.bgGlow}
        />
      </View>

      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.kb}>
        <View style={styles.container}>
          <View style={styles.topRow}>
            <View style={{ flex: 1 }}>
              <ThemedText style={[styles.kicker, { color: palette.muted2 }]}>Cold start</ThemedText>
              <ThemedText style={[styles.title, { color: palette.text }]}>Pick your vibe</ThemedText>
              <ThemedText style={[styles.subtitle, { color: palette.muted }]}>
                Choose {MIN_SEEDS}-{MAX_SEEDS} songs you love. We’ll personalize everything around it.
              </ThemedText>
            </View>
            <TouchableOpacity
              accessibilityRole="button"
              onPress={logout}
              style={[styles.logoutPill, { backgroundColor: palette.glass, borderColor: palette.stroke }]}
            >
              <Ionicons name="log-out-outline" size={16} color={palette.muted} />
              <ThemedText style={[styles.logoutText, { color: palette.muted }]}>{username ? 'Sign out' : 'Sign out'}</ThemedText>
            </TouchableOpacity>
          </View>

          {step === 'pick' ? (
            <>
              <View style={[styles.searchWrap, { backgroundColor: palette.glass, borderColor: palette.stroke }]}>
                <Ionicons name="search" size={18} color={palette.muted2} />
                <TextInput
                  placeholder="Search songs in the catalogue…"
                  placeholderTextColor={palette.muted2}
                  style={[styles.searchInput, { color: palette.text }, ({ outlineStyle: 'none' } as any)]}
                  value={query}
                  onChangeText={setQuery}
                  autoCorrect={false}
                  returnKeyType="search"
                />
                {query.length > 0 ? (
                  <Pressable onPress={() => setQuery('')} hitSlop={10}>
                    <Ionicons name="close-circle" size={18} color={palette.muted2} />
                  </Pressable>
                ) : (
                  <Ionicons name="sparkles-outline" size={18} color={palette.muted2} />
                )}
              </View>

              <View style={[styles.chipsWrap, { borderColor: palette.stroke }]}>
                <View style={styles.chipsHeader}>
                  <ThemedText style={[styles.chipsTitle, { color: palette.text }]}>Selected</ThemedText>
                  <ThemedText style={[styles.chipsCount, { color: palette.muted }]}>{seeds.length}/{MAX_SEEDS}</ThemedText>
                </View>
                <View style={styles.chipsRow}>
                  {seeds.length === 0 ? (
                    <ThemedText style={[styles.helper, { color: palette.muted }]}>No picks yet. Add at least {MIN_SEEDS}.</ThemedText>
                  ) : (
                    seeds.map((sid) => (
                      <TouchableOpacity key={sid} onPress={() => removeSeed(sid)} style={[styles.chip, { backgroundColor: 'rgba(255,255,255,0.06)', borderColor: palette.stroke }]}>
                        <ThemedText numberOfLines={1} style={[styles.chipText, { color: palette.text }]}>{sid}</ThemedText>
                        <Ionicons name="close" size={14} color={palette.muted2} />
                      </TouchableOpacity>
                    ))
                  )}
                </View>
              </View>

              {searchError ? <ThemedText style={styles.errorText}>{searchError}</ThemedText> : null}
              {searchLoading ? <ActivityIndicator style={{ marginTop: 10 }} /> : null}

              <FlatList
                data={results}
                keyExtractor={(item) => item.song_id}
                keyboardShouldPersistTaps="handled"
                style={{ flex: 1 }}
                contentContainerStyle={{ gap: 12, paddingTop: 12, paddingBottom: 14 }}
                renderItem={({ item }) => {
                  const added = seeds.includes(item.song_id);
                  const disabled = !added && seeds.length >= MAX_SEEDS;
                  return (
                    <View style={[styles.row, { backgroundColor: palette.glass, borderColor: palette.stroke }]}>
                      <Image source={{ uri: item.cover_url || FALLBACK_COVER }} style={styles.cover} contentFit="cover" transition={200} />
                      <View style={{ flex: 1, gap: 2 }}>
                        <ThemedText numberOfLines={1} style={[styles.rowTitle, { color: palette.text }]}>{item.title}</ThemedText>
                        <ThemedText numberOfLines={1} style={[styles.rowSub, { color: palette.muted }]}>{item.artist}</ThemedText>
                        {!!item.album && <ThemedText numberOfLines={1} style={[styles.rowMeta, { color: palette.muted2 }]}>{item.album}</ThemedText>}
                      </View>
                      <View style={styles.rowActions}>
                        <Pressable
                          onPress={() => toggleLikeSong(item.song_id)}
                          style={[styles.iconBtn, isLiked(item.song_id) ? styles.iconBtnLikeActive : styles.iconBtnGhost]}
                        >
                          <Ionicons name={isLiked(item.song_id) ? 'heart' : 'heart-outline'} size={18} color={isLiked(item.song_id) ? '#fff' : palette.muted} />
                        </Pressable>
                        <Pressable
                          disabled={disabled}
                          onPress={() => addSeed(item.song_id)}
                          style={[
                            styles.iconBtn,
                            added ? styles.iconBtnActive : styles.iconBtnPrimary,
                            disabled && { opacity: 0.5 },
                          ]}
                        >
                          <Ionicons name={added ? 'checkmark' : 'add'} size={18} color={added ? '#111' : '#111'} />
                        </Pressable>
                      </View>
                    </View>
                  );
                }}
                ListEmptyComponent={
                  query.trim() ? (
                    <ThemedText style={[styles.helper, { color: palette.muted }]}>No matches in the catalogue.</ThemedText>
                  ) : (
                    <ThemedText style={[styles.helper, { color: palette.muted }]}>Search and add your first picks.</ThemedText>
                  )
                }
              />

              {recError ? <ThemedText style={styles.errorText}>{recError}</ThemedText> : null}

              <Pressable
                accessibilityRole="button"
                disabled={seeds.length < MIN_SEEDS || recLoading}
                onPress={submit}
                style={({ pressed }) => [
                  styles.ctaWrap,
                  (pressed && seeds.length >= MIN_SEEDS && !recLoading) && styles.ctaPressed,
                  (seeds.length < MIN_SEEDS || recLoading) && styles.ctaDisabled,
                ]}
              >
                <LinearGradient colors={['#1DB954', '#18a34a']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.cta}>
                  {recLoading ? (
                    <ActivityIndicator color="#041007" />
                  ) : (
                    <View style={styles.ctaInner}>
                      <ThemedText style={styles.ctaText}>Continue</ThemedText>
                      <ThemedText style={styles.ctaMeta}>{seeds.length}/{MIN_SEEDS} minimum</ThemedText>
                    </View>
                  )}
                </LinearGradient>
              </Pressable>
            </>
          ) : (
            <>
              <View style={styles.resultsHeader}>
                <ThemedText style={[styles.resultsTitle, { color: palette.text }]}>You might like</ThemedText>
                <ThemedText style={[styles.resultsSub, { color: palette.muted }]}>Instant picks from your vibe</ThemedText>
              </View>

              {recError ? <ThemedText style={styles.errorText}>{recError}</ThemedText> : null}

              <FlatList
                data={recs}
                keyExtractor={(item) => item.song_id}
                style={{ flex: 1 }}
                contentContainerStyle={{ gap: 12, paddingTop: 10, paddingBottom: 14 }}
                renderItem={({ item }) => (
                  <View style={[styles.row, { backgroundColor: palette.glass, borderColor: palette.stroke }]}>
                    <Image source={{ uri: item.cover_url || FALLBACK_COVER }} style={styles.cover} contentFit="cover" transition={200} />
                    <View style={{ flex: 1, gap: 2 }}>
                      <ThemedText numberOfLines={1} style={[styles.rowTitle, { color: palette.text }]}>{item.title}</ThemedText>
                      <ThemedText numberOfLines={1} style={[styles.rowSub, { color: palette.muted }]}>{item.artist}</ThemedText>
                    </View>
                    <View style={styles.rowActions}>
                      <Pressable
                        onPress={() => toggleLikeSong(item.song_id)}
                        style={[styles.iconBtn, isLiked(item.song_id) ? styles.iconBtnLikeActive : styles.iconBtnGhost]}
                      >
                        <Ionicons name={isLiked(item.song_id) ? 'heart' : 'heart-outline'} size={18} color={isLiked(item.song_id) ? '#fff' : palette.muted} />
                      </Pressable>
                    </View>
                  </View>
                )}
              />

              <Pressable accessibilityRole="button" onPress={finish} style={({ pressed }) => [styles.ctaWrap, pressed && styles.ctaPressed]}>
                <LinearGradient colors={['#1DB954', '#18a34a']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.cta}>
                  <View style={styles.ctaInner}>
                    <ThemedText style={styles.ctaText}>Finish</ThemedText>
                    <Ionicons name="arrow-forward" size={18} color="#041007" />
                  </View>
                </LinearGradient>
              </Pressable>
            </>
          )}
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  page: {
    flex: 1,
    backgroundColor: '#050606',
    overflow: 'hidden',
  },
  kb: {
    flex: 1,
  },
  bgGlowWrap: {
    position: 'absolute',
    left: -120,
    right: -120,
    top: -160,
    height: 520,
  },
  bgGlow: {
    flex: 1,
    borderRadius: 340,
    transform: [{ rotateZ: '-12deg' }],
  },
  container: {
    width: '100%',
    maxWidth: 720,
    alignSelf: 'center',
    flex: 1,
    minHeight: 0,
    paddingHorizontal: 18,
    paddingVertical: 20,
    gap: 12,
  },
  topRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },
  kicker: {
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.7,
    textTransform: 'uppercase',
  },
  title: {
    fontSize: 34,
    fontWeight: '900',
    letterSpacing: -0.5,
    marginTop: 2,
  },
  subtitle: {
    fontSize: 14,
    lineHeight: 20,
    marginTop: 6,
  },
  logoutPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
  },
  logoutText: {
    fontWeight: '800',
    fontSize: 12,
  },
  searchWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderWidth: 1,
    marginTop: 6,
  },
  searchInput: {
    flex: 1,
    fontSize: 15,
    paddingVertical: 0,
  },
  chipsWrap: {
    marginTop: 10,
    borderRadius: 16,
    borderWidth: 1,
    padding: 12,
    gap: 10,
    backgroundColor: 'rgba(0,0,0,0.16)',
  },
  chipsHeader: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
  },
  chipsTitle: {
    fontWeight: '900',
    fontSize: 14,
  },
  chipsCount: {
    fontWeight: '800',
    fontSize: 12,
  },
  chipsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  chip: {
    maxWidth: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
  },
  chipText: {
    maxWidth: 240,
    fontWeight: '700',
  },
  helper: {
    opacity: 0.9,
  },
  errorText: {
    color: '#fb7185',
    fontWeight: '700',
    marginTop: 6,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    borderRadius: 16,
    padding: 12,
    borderWidth: 1,
  },
  cover: {
    width: 54,
    height: 54,
    borderRadius: 12,
  },
  rowTitle: {
    fontWeight: '900',
  },
  rowSub: {
    fontSize: 13,
    fontWeight: '700',
    opacity: 0.95,
  },
  rowMeta: {
    fontSize: 12,
  },
  rowActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  iconBtn: {
    width: 40,
    height: 40,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBtnGhost: {
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.14)',
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  iconBtnPrimary: {
    backgroundColor: '#1DB954',
  },
  iconBtnActive: {
    backgroundColor: 'rgba(29,185,84,0.25)',
    borderWidth: 1,
    borderColor: 'rgba(29,185,84,0.55)',
  },
  iconBtnLikeActive: {
    backgroundColor: '#e11d48',
  },
  ctaWrap: {
    borderRadius: 18,
    overflow: 'hidden',
    marginTop: 10,
    shadowColor: '#000',
    shadowOpacity: 0.22,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 12 },
  },
  ctaPressed: {
    transform: [{ scale: 0.99 }],
  },
  ctaDisabled: {
    opacity: 0.55,
  },
  cta: {
    paddingVertical: 14,
    paddingHorizontal: 14,
  },
  ctaInner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  ctaText: {
    color: '#041007',
    fontWeight: '900',
    letterSpacing: 0.2,
  },
  ctaMeta: {
    color: 'rgba(4,16,7,0.70)',
    fontWeight: '900',
    fontSize: 12,
  },
  resultsHeader: {
    marginTop: 8,
    gap: 4,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: '900',
  },
  resultsSub: {
    fontSize: 13,
    fontWeight: '700',
  },
});

