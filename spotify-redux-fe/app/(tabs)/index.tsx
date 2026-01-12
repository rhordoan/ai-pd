import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, FlatList, Image, StyleSheet, TextInput, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated, { useSharedValue, withRepeat, withTiming, useAnimatedStyle, interpolateColor } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import { ThemedText } from '@/components/themed-text';
import { Section } from '@/components/home/section';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { useThemeColor } from '@/hooks/use-theme-color';
import { MiniPlayer, type TrackInfo, type QueueItem } from '@/components/player/mini-player';
import {
  getColdStartRecommendations,
  getSimilarRecommendations,
  getCategories,
  getSeeds,
  getLikes,
  toggleLike,
  searchTracks,
  type CategoriesResponse,
  type RecommendationItem,
  type SearchResult,
} from '@/lib/api';
import { useAuth } from '@/providers/auth';
import { useRouter } from 'expo-router';
import { storageDelete, storageGet, storageSet } from '@/lib/storage';

const MIN_SEEDS = 3;
const MAX_SEEDS = 5;
const FALLBACK_COVER = 'https://via.placeholder.com/120x120.png?text=Track';

// (old BackgroundCanvas removed in favor of Skia MetaballsBackground)

export default function HomeScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const isDark = colorScheme === 'dark';
  const { token, username, logout, ready: authReady } = useAuth();
  const router = useRouter();
  const hue = useSharedValue(0);
  const baseTextColor = useThemeColor({}, 'text');
  const [currentTrack, setCurrentTrack] = React.useState<TrackInfo | null>(null);
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchFocused, setSearchFocused] = useState(false);
  const [seeds, setSeeds] = useState<string[]>([]);
  const [savedSeeds, setSavedSeeds] = useState<string[] | null>(null);
  const [savedSeedsLoaded, setSavedSeedsLoaded] = useState(false);
  const [recs, setRecs] = useState<RecommendationItem[]>([]);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const [coldStartJustCompleted, setColdStartJustCompleted] = useState(false);
  const [categories, setCategories] = useState<CategoriesResponse | null>(null);
  const [categoriesLoading, setCategoriesLoading] = useState(false);
  const [categoriesError, setCategoriesError] = useState<string | null>(null);
  const categoriesFetchRef = React.useRef<{ lastKey: string }>({ lastKey: '' });
  const [queue, setQueue] = useState<RecommendationItem[]>([]);
  const [queueLoading, setQueueLoading] = useState(false);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [autoQueue, setAutoQueue] = useState(true);
  const [history, setHistory] = useState<RecommendationItem[]>([]);
  const [likes, setLikes] = useState<Set<string>>(new Set());
  const insets = useSafeAreaInsets();

  // Restore last playing track on app refresh/relaunch
  useEffect(() => {
    (async () => {
      const raw = await storageGet('player_current_track');
      if (!raw) return;
      try {
        const parsed = JSON.parse(raw) as TrackInfo;
        if (parsed?.title) setCurrentTrack(parsed);
      } catch {
        // ignore
      }
    })();
  }, []);

  // Persist current track (do not persist position)
  useEffect(() => {
    if (!currentTrack) {
      storageDelete('player_current_track');
      return;
    }
    storageSet('player_current_track', JSON.stringify(currentTrack));
  }, [currentTrack]);
  React.useEffect(() => {
    hue.value = withRepeat(withTiming(1, { duration: 8000 }), -1, true);
  }, [hue]);

  const greetingStyle = useAnimatedStyle(() => ({
    color: interpolateColor(
      hue.value,
      [0, 0.5, 1],
      [isDark ? '#FFFFFF' : '#111111', '#1DB954', isDark ? '#FFFFFF' : '#111111']
    ),
  }));

  // Debounced search to avoid chatty network calls
  useEffect(() => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    setSearchLoading(true);
    setSearchError(null);
    const handle = setTimeout(() => {
      searchTracks(query)
        .then((res) => setSearchResults(res))
        .catch((err: Error) => setSearchError(err.message))
        .finally(() => setSearchLoading(false));
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  const addSeed = (songId: string) => {
    setSeeds((prev) => {
      if (prev.includes(songId) || prev.length >= MAX_SEEDS) return prev;
      return [...prev, songId];
    });
  };

  const removeSeed = (songId: string) => {
    setSeeds((prev) => prev.filter((s) => s !== songId));
  };

  const fetchRecommendations = async () => {
    setRecLoading(true);
    setRecError(null);
    try {
      const res = await getColdStartRecommendations(seeds, 10, token ?? undefined);
      setRecs(res.results);
      setColdStartJustCompleted(true);
      // If authenticated, seeds are persisted server-side; update local view immediately
      if (token) {
        setSavedSeeds([...seeds]);
        setSavedSeedsLoaded(true);
      }
      // Clear picker chips after success to keep UI clean
      setSeeds([]);
    } catch (err) {
      setRecError((err as Error).message);
    } finally {
      setRecLoading(false);
    }
  };

  const fetchSimilar = async (songId: string) => {
    setRecLoading(true);
    setRecError(null);
    try {
      const res = await getSimilarRecommendations(songId, 8, token ?? undefined);
      setRecs(res.results);
    } catch (err) {
      setRecError((err as Error).message);
    } finally {
      setRecLoading(false);
    }
  };

  const handlePlay = (item: {
    title: string;
    artist: string;
    preview?: string | null;
    song_id?: string;
    songId?: string;
    cover_url?: string | null;
    coverUrl?: string | null;
  }) => {
    setCurrentTrack({
      title: item.title,
      artist: item.artist,
      uri: item.preview ?? 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
      songId: item.songId ?? item.song_id,
      coverUrl: (item.coverUrl ?? item.cover_url ?? undefined) || undefined,
    });
  };

  const isLiked = (songId: string | undefined) => (songId ? likes.has(songId) : false);

  const toggleLikeSong = async (songId: string) => {
    if (!songId) return;
    if (!token) {
      Alert.alert('Login required', 'Log in to like songs.');
      router.push('/login');
      return;
    }
    // optimistic update
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
    } catch (err) {
      // revert on error
      setLikes((prev) => {
        const next = new Set(prev);
        if (next.has(songId)) next.delete(songId);
        else next.add(songId);
        return next;
      });
      Alert.alert('Error', (err as Error).message || 'Failed to update like');
    }
  };

  // Fetch stored seeds for logged in users
  useEffect(() => {
    if (!authReady) return;

    // Token changed (login/logout/new account): reset seed state so UI can't reuse previous account's seeds
    setSavedSeeds(null);
    setSavedSeedsLoaded(false);
    setColdStartJustCompleted(false);
    setRecs([]);

    if (!token) {
      setSavedSeedsLoaded(true);
      setLikes(new Set());
      return;
    }
    getSeeds(token)
      .then((res) => setSavedSeeds(res.seeds))
      .catch(() => setSavedSeeds(null))
      .finally(() => setSavedSeedsLoaded(true));

    getLikes(token)
      .then((res) => setLikes(new Set(res.likes)))
      .catch(() => setLikes(new Set()));
  }, [token, authReady]);

  // One-time cold start: if authenticated but not seeded, send user to onboarding screen
  useEffect(() => {
    if (!authReady) return;
    if (!token) return;
    if (!savedSeedsLoaded) return;
    if (!savedSeeds || savedSeeds.length === 0) {
      router.replace('/onboarding');
    }
  }, [authReady, token, savedSeedsLoaded, savedSeeds, router]);

  // Fetch categories when seeds change or savedSeeds available
  useEffect(() => {
    if (!authReady) return;
    // If logged in, wait until we know whether seeds exist (avoids extra fetches)
    if (token && !savedSeedsLoaded) return;
    // If not logged in, only fetch categories once user has at least one seed
    if (!token && seeds.length === 0) return;

    const seed = (savedSeeds && savedSeeds[0]) || seeds[0];
    const key = `${token ?? 'anon'}|${seed ?? ''}|8`;
    if (categoriesFetchRef.current.lastKey === key) return;
    categoriesFetchRef.current.lastKey = key;

    setCategoriesLoading(true);
    setCategoriesError(null);
    getCategories(token ?? undefined, seed ?? undefined, 8)
      .then((res) => {
        // ignore stale responses if key changed
        if (categoriesFetchRef.current.lastKey !== key) return;
        setCategories(res);
      })
      .catch((err: Error) => {
        if (categoriesFetchRef.current.lastKey !== key) return;
        setCategoriesError(err.message);
      })
      .finally(() => {
        if (categoriesFetchRef.current.lastKey !== key) return;
        setCategoriesLoading(false);
      });
  }, [token, savedSeeds, seeds, authReady, savedSeedsLoaded]);

  const loadQueueForSong = async (songId: string) => {
    setQueueLoading(true);
    setQueueError(null);
    try {
      const res = await getSimilarRecommendations(songId, 8, token ?? undefined);
      setQueue(res.results);
    } catch (err) {
      setQueueError((err as Error).message);
    } finally {
      setQueueLoading(false);
    }
  };

  const playNextFromQueue = () => {
    if (!queue.length) return;
    const [next, ...rest] = queue;
    // push current into history
    if (currentTrack?.songId) {
      setHistory((h) => [
        {
          song_id: currentTrack.songId,
          title: currentTrack.title ?? '',
          artist: currentTrack.artist ?? '',
          cover_url: (currentTrack as any).cover_url ?? undefined,
        } as RecommendationItem,
        ...h,
      ]);
    }
    handlePlay({
      title: next.title,
      artist: next.artist,
      preview: undefined,
      song_id: next.song_id,
      cover_url: next.cover_url,
    });
    setQueue(rest);
  };

  const playPrevFromHistory = () => {
    if (!history.length) return;
    const [prev, ...rest] = history;
    if (currentTrack?.songId) {
      setQueue((q) => [
        {
          song_id: currentTrack.songId,
          title: currentTrack.title ?? '',
          artist: currentTrack.artist ?? '',
          cover_url: (currentTrack as any).cover_url ?? undefined,
        } as RecommendationItem,
        ...q,
      ]);
    }
    handlePlay({
      title: prev.title,
      artist: prev.artist,
      preview: undefined,
      song_id: (prev as any).song_id ?? (prev as any).songId,
      cover_url: (prev as any).cover_url ?? undefined,
    });
    setHistory(rest);
  };

  const selectQueueItem = (songId: string) => {
    const idx = queue.findIndex((q) => q.song_id === songId);
    if (idx < 0) return;
    const chosen = queue[idx];
    // push current into history
    if (currentTrack?.songId) {
      setHistory((h) => [
        {
          song_id: currentTrack.songId,
          title: currentTrack.title ?? '',
          artist: currentTrack.artist ?? '',
          cover_url: (currentTrack as any).cover_url ?? undefined,
        } as RecommendationItem,
        ...h,
      ]);
    }
    handlePlay({
      title: chosen.title,
      artist: chosen.artist,
      preview: undefined,
      song_id: chosen.song_id,
      cover_url: chosen.cover_url,
    });
    setQueue((q) => q.filter((it) => it.song_id !== songId));
  };

  // When current track changes and has a songId, load a queue
  useEffect(() => {
    if (currentTrack?.songId && autoQueue) {
      loadQueueForSong(currentTrack.songId);
    }
  }, [currentTrack?.songId, autoQueue]);

  return (
    <View style={styles.page}>
      {/* Premium background to match login/onboarding */}
      <LinearGradient
        colors={isDark ? ['#050606', '#090f0c', '#050606'] : ['#f7faf8', '#eef7f2', '#f5f7fa']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFill}
      />
      <View style={styles.bgGlowWrap} pointerEvents="none">
        <LinearGradient
          colors={['rgba(29,185,84,0.20)', 'rgba(88,86,214,0.14)', 'rgba(255,255,255,0.00)']}
          start={{ x: 0.1, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.bgGlow}
        />
      </View>
      <Animated.ScrollView contentContainerStyle={styles.contentContainer} scrollEventThrottle={16}>
        <Section title="Search music">
          <LinearGradient
            colors={
              isDark
                ? ['rgba(255,255,255,0.10)', 'rgba(255,255,255,0.04)']
                : ['rgba(255,255,255,0.98)', 'rgba(245,247,250,0.94)']
            }
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={[styles.searchGradient, searchFocused && styles.searchGradientFocused]}
          >
            <View style={[styles.searchInner, searchFocused && styles.searchInnerFocused]}>
              <Ionicons name="search" size={18} color={isDark ? '#d6d6d6' : '#4b5563'} />
              <TextInput
                placeholder="What do you want to listen to?"
                placeholderTextColor={isDark ? 'rgba(255,255,255,0.55)' : 'rgba(17,17,17,0.45)'}
                style={[styles.searchInput, { color: isDark ? '#fff' : '#111' }, ({ outlineStyle: 'none' } as any)]}
                value={query}
                onChangeText={setQuery}
                autoCorrect={false}
                onFocus={() => setSearchFocused(true)}
                onBlur={() => setSearchFocused(false)}
                returnKeyType="search"
                underlineColorAndroid="transparent"
                selectionColor="#1DB954"
              />
              {query.length > 0 ? (
                <TouchableOpacity accessibilityRole="button" accessibilityLabel="Clear search" onPress={() => setQuery('')}>
                  <Ionicons name="close-circle" size={18} color={isDark ? 'rgba(255,255,255,0.55)' : 'rgba(17,17,17,0.45)'} />
                </TouchableOpacity>
              ) : (
                <Ionicons name="mic-outline" size={18} color={isDark ? 'rgba(255,255,255,0.45)' : 'rgba(17,17,17,0.35)'} />
              )}
            </View>
          </LinearGradient>
          {searchError && <ThemedText style={styles.errorText}>{searchError}</ThemedText>}
          {searchLoading && <ActivityIndicator style={{ marginVertical: 8 }} />}
          {!searchLoading && (
            <FlatList
              data={searchResults}
              keyExtractor={(item) => item.song_id}
              keyboardShouldPersistTaps="handled"
              contentContainerStyle={{ gap: 12 }}
              renderItem={({ item }) => (
                <TouchableOpacity
                  activeOpacity={0.85}
                  style={[styles.searchResultCard, isDark ? styles.searchResultCardDark : styles.searchResultCardLight]}
                  onPress={() =>
                    handlePlay({
                      title: item.title,
                      artist: item.artist,
                      preview: item.preview_url,
                      song_id: item.song_id,
                      cover_url: item.cover_url,
                    })
                  }
                >
                  <View style={styles.searchResultLeft}>
                    <View style={styles.searchCoverWrap}>
                      <Image source={{ uri: item.cover_url || FALLBACK_COVER }} style={styles.searchCover} resizeMode="cover" />
                    </View>
                    <View style={styles.searchResultMeta}>
                      <ThemedText numberOfLines={1} style={styles.resultTitle}>{item.title}</ThemedText>
                      <ThemedText numberOfLines={1} style={styles.resultArtist}>{item.artist}</ThemedText>
                      {item.album ? <ThemedText numberOfLines={1} style={styles.resultAlbum}>{item.album}</ThemedText> : null}
                    </View>
                  </View>

                  <View style={styles.searchResultActions}>
                    <TouchableOpacity
                      accessibilityRole="button"
                      accessibilityLabel="Get similar"
                      style={[styles.iconButtonRound, styles.iconButtonGhost]}
                      onPress={() => fetchSimilar(item.song_id)}
                    >
                      <Ionicons name="sparkles-outline" size={18} color={isDark ? '#fff' : '#111'} />
                    </TouchableOpacity>

                    <TouchableOpacity
                      accessibilityRole="button"
                      accessibilityLabel={seeds.includes(item.song_id) ? 'Added to vibe' : 'Add to vibe'}
                      style={[
                        styles.iconButtonRound,
                        seeds.includes(item.song_id) ? styles.iconButtonRoundActive : styles.iconButtonRoundPrimary,
                        (seeds.length >= MAX_SEEDS && !seeds.includes(item.song_id)) && styles.iconButtonDisabled,
                      ]}
                      disabled={seeds.length >= MAX_SEEDS && !seeds.includes(item.song_id)}
                      onPress={() => addSeed(item.song_id)}
                    >
                      <Ionicons
                        name={seeds.includes(item.song_id) ? 'checkmark' : 'add'}
                        size={18}
                        color={seeds.includes(item.song_id) ? '#111' : '#111'}
                      />
                    </TouchableOpacity>

                    <TouchableOpacity
                      accessibilityRole="button"
                      accessibilityLabel={isLiked(item.song_id) ? 'Unlike' : 'Like'}
                      style={[
                        styles.iconButtonRound,
                        isLiked(item.song_id) ? styles.iconButtonRoundLikeActive : styles.iconButtonGhost,
                      ]}
                      onPress={() => toggleLikeSong(item.song_id)}
                    >
                      <Ionicons
                        name={isLiked(item.song_id) ? 'heart' : 'heart-outline'}
                        size={18}
                        color={isLiked(item.song_id) ? '#fff' : (isDark ? '#fff' : '#111')}
                      />
                    </TouchableOpacity>
                  </View>
                </TouchableOpacity>
              )}
              ListEmptyComponent={
                query.trim() ? (
                  <ThemedText style={styles.helperText}>No results yet</ThemedText>
                ) : (
                  <ThemedText style={styles.helperText}>Start typing to search</ThemedText>
                )
              }
            />
          )}
        </Section>

        <View style={[styles.heroCard, isDark ? styles.heroCardDark : styles.heroCardLight]}>
          <LinearGradient
            colors={isDark ? ['rgba(255,255,255,0.06)', 'rgba(255,255,255,0.00)'] : ['rgba(29,185,84,0.10)', 'rgba(255,255,255,0.00)']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.heroGlow}
          />
          <View style={styles.greetingRow}>
            <View>
              <Animated.Text style={[styles.greetingText, { color: baseTextColor }, greetingStyle]}>
                Good {getDaytimeGreeting()}
              </Animated.Text>
              <ThemedText style={styles.heroSub}>{username ? `Welcome, ${username}` : 'Sign in to sync your taste.'}</ThemedText>
            </View>
            <View style={styles.greetingActions}>
              {!token ? (
                <TouchableOpacity style={[styles.loginPill, styles.loginPillPrimary]} onPress={() => router.push('/login')}>
                  <IconSymbol name="person.fill" size={16} color="#111" />
                  <ThemedText style={styles.loginPillText}>Log in</ThemedText>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity style={[styles.loginPill, styles.loginPillGhost]} onPress={logout}>
                  <IconSymbol name="checkmark.seal.fill" size={16} color={isDark ? '#fff' : '#111'} />
                  <ThemedText style={[styles.loginPillText, styles.loginPillTextGhost]}>Sign out</ThemedText>
                </TouchableOpacity>
              )}
            </View>
          </View>
          <ThemedText style={styles.heroSub}>Handpicked mixes and fresh drops tuned to your mood.</ThemedText>
        </View>

        {!token && (!savedSeeds || savedSeeds.length === 0 || coldStartJustCompleted) ? (
          <Section title="Pick your vibe (cold start)">
            <ThemedText style={styles.helperText}>Select {MIN_SEEDS}-{MAX_SEEDS} songs you like. We’ll recommend similar songs from the model catalogue and fetch covers via iTunes.</ThemedText>
            <View style={styles.seedsRow}>
              {seeds.map((seed) => (
                <TouchableOpacity key={seed} style={styles.seedChip} onPress={() => removeSeed(seed)}>
                  <ThemedText numberOfLines={1} style={styles.seedText}>{seed}</ThemedText>
                  <ThemedText style={styles.seedRemove}>✕</ThemedText>
                </TouchableOpacity>
              ))}
              {seeds.length === 0 && <ThemedText style={styles.helperText}>No seeds yet</ThemedText>}
            </View>
            <TouchableOpacity
              style={[
                styles.actionButton,
                (seeds.length < MIN_SEEDS || recLoading) && styles.actionButtonDisabled,
              ]}
              disabled={seeds.length < MIN_SEEDS || recLoading}
              onPress={fetchRecommendations}
            >
              {recLoading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <ThemedText style={styles.actionButtonText}>Get recommendations</ThemedText>
              )}
            </TouchableOpacity>
            {recError && <ThemedText style={styles.errorText}>{recError}</ThemedText>}

            {recs.length > 0 && (
              <View style={{ marginTop: 14, gap: 10 }}>
                <View style={styles.mightLikeHeader}>
                  <ThemedText style={styles.mightLikeTitle}>You might like</ThemedText>
                  <ThemedText style={styles.mightLikeSub}>Based on your picks</ThemedText>
                </View>
                <FlatList
                  data={recs}
                  keyExtractor={(item) => item.song_id}
                  scrollEnabled={false}
                  contentContainerStyle={{ gap: 12 }}
                  renderItem={({ item }) => (
                    <TouchableOpacity
                      style={[styles.resultRow, isDark ? styles.resultRowDark : styles.resultRowLight]}
                      onPress={() =>
                        handlePlay({
                          title: item.title,
                          artist: item.artist,
                          preview: undefined,
                          song_id: item.song_id,
                          cover_url: item.cover_url,
                        })
                      }
                    >
                      <Image source={{ uri: item.cover_url || FALLBACK_COVER }} style={styles.coverThumb} />
                      <View style={styles.resultMeta}>
                        <ThemedText numberOfLines={1} style={styles.resultTitle}>{item.title}</ThemedText>
                        <ThemedText numberOfLines={1} style={styles.resultArtist}>{item.artist}</ThemedText>
                      </View>
                      <TouchableOpacity
                        style={[
                          styles.smallButton,
                          isLiked(item.song_id) ? styles.likeButtonActive : styles.likeButton,
                        ]}
                        onPress={() => toggleLikeSong(item.song_id)}
                      >
                        <Ionicons
                          name={isLiked(item.song_id) ? 'heart' : 'heart-outline'}
                          size={18}
                          color={isLiked(item.song_id) ? '#fff' : '#111'}
                        />
                      </TouchableOpacity>
                    </TouchableOpacity>
                  )}
                />
              </View>
            )}
          </Section>
        ) : null}

        <Section title="For you">
          {categoriesLoading && <ActivityIndicator style={{ marginVertical: 8 }} />}
          {categoriesError && <ThemedText style={styles.errorText}>{categoriesError}</ThemedText>}
          {!categoriesLoading && categories?.buckets.map((bucket) => (
            <View key={bucket.label} style={{ marginBottom: 12, gap: 8 }}>
              <View style={styles.bucketHeader}>
                <ThemedText style={styles.bucketLabel}>{bucket.label}</ThemedText>
                {bucket.seed_used ? <ThemedText style={styles.bucketSeed}>Based on {bucket.seed_used}</ThemedText> : null}
              </View>
              <FlatList
                data={bucket.items}
                keyExtractor={(item) => `${bucket.label}-${item.song_id}`}
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.carouselContent}
                renderItem={({ item }) => (
                  <TouchableOpacity
                    style={[styles.cardItem, isDark ? styles.resultRowDark : styles.resultRowLight]}
                    onPress={() =>
                      handlePlay({
                        title: item.title,
                        artist: item.artist,
                        preview: undefined,
                        song_id: item.song_id,
                        cover_url: item.cover_url,
                      })
                    }
                  >
                    <View style={styles.cardImageWrap}>
                      <Image source={{ uri: item.cover_url || FALLBACK_COVER }} style={styles.bucketCover} />
                      <TouchableOpacity
                        style={[
                          styles.likeFloating,
                          isLiked(item.song_id) ? styles.likeButtonActive : styles.likeButton,
                        ]}
                        onPress={() => toggleLikeSong(item.song_id)}
                      >
                        <Ionicons
                          name={isLiked(item.song_id) ? 'heart' : 'heart-outline'}
                          size={16}
                          color={isLiked(item.song_id) ? '#fff' : '#111'}
                        />
                      </TouchableOpacity>
                    </View>
                    <ThemedText numberOfLines={1} style={styles.resultTitle}>{item.title}</ThemedText>
                    <ThemedText numberOfLines={1} style={styles.resultArtist}>{item.artist}</ThemedText>
                  </TouchableOpacity>
                )}
              />
            </View>
          ))}
        </Section>

        <Section title="Up next">
          {queueLoading && <ActivityIndicator style={{ marginVertical: 8 }} />}
          {queueError && <ThemedText style={styles.errorText}>{queueError}</ThemedText>}
          {!queueLoading && queue.length === 0 && <ThemedText style={styles.helperText}>Start a song to see the queue.</ThemedText>}
          {!queueLoading && queue.length > 0 && (
            <FlatList
              data={queue}
              keyExtractor={(item) => item.song_id}
              scrollEnabled={false}
              contentContainerStyle={{ gap: 12 }}
              renderItem={({ item, index }) => (
                <TouchableOpacity
                  style={[styles.resultRow, isDark ? styles.resultRowDark : styles.resultRowLight]}
                  onPress={() =>
                    handlePlay({
                      title: item.title,
                      artist: item.artist,
                      preview: undefined,
                      song_id: item.song_id,
                      cover_url: item.cover_url,
                    })
                  }
                >
                  <View style={styles.queueIndex}>
                    <ThemedText style={styles.queueIndexText}>{index + 1}</ThemedText>
                  </View>
                  <Image source={{ uri: item.cover_url || FALLBACK_COVER }} style={styles.coverThumb} />
                  <View style={styles.resultMeta}>
                    <ThemedText numberOfLines={1} style={styles.resultTitle}>{item.title}</ThemedText>
                    <ThemedText numberOfLines={1} style={styles.resultArtist}>{item.artist}</ThemedText>
                  </View>
                  <TouchableOpacity
                    style={[
                      styles.smallButton,
                      isLiked(item.song_id) ? styles.likeButtonActive : styles.likeButton,
                    ]}
                    onPress={() => toggleLikeSong(item.song_id)}
                  >
                    <Ionicons
                      name={isLiked(item.song_id) ? 'heart' : 'heart-outline'}
                      size={18}
                      color={isLiked(item.song_id) ? '#fff' : '#111'}
                    />
                  </TouchableOpacity>
                </TouchableOpacity>
              )}
            />
          )}
          {queue.length > 0 && (
            <View style={{ flexDirection: 'row', gap: 10, marginTop: 10 }}>
            <TouchableOpacity
              style={[styles.actionButton, history.length === 0 && styles.actionButtonDisabled, { flex: 1 }]}
              disabled={history.length === 0}
              onPress={playPrevFromHistory}
            >
              <ThemedText style={styles.actionButtonText}>Previous</ThemedText>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionButton, queue.length === 0 && styles.actionButtonDisabled, { flex: 1 }]}
              disabled={queue.length === 0}
              onPress={playNextFromQueue}
            >
              <ThemedText style={styles.actionButtonText}>Next</ThemedText>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionButton, { flex: 1 }]}
              onPress={() => currentTrack?.songId && loadQueueForSong(currentTrack.songId)}
            >
              <ThemedText style={styles.actionButtonText}>Refresh queue</ThemedText>
            </TouchableOpacity>
            </View>
          )}
        </Section>
      </Animated.ScrollView>
      <View
        style={[
          styles.floatingPlayer,
          {
            paddingBottom: Math.max(insets.bottom + 10, 18),
            paddingHorizontal: 14,
            bottom: 10,
          },
        ]}
        pointerEvents="box-none"
      >
        <LinearGradient
          colors={
            isDark
              ? ['rgba(10,10,10,0.78)', 'rgba(10,10,10,0.68)']
              : ['rgba(255,255,255,0.9)', 'rgba(255,255,255,0.8)']
          }
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.playerBackdrop}
        />
        <MiniPlayer
          track={currentTrack}
          isLiked={isLiked(currentTrack?.songId)}
          onToggleLike={(id) => id && toggleLikeSong(id)}
          onNext={playNextFromQueue}
          onPrev={playPrevFromHistory}
          hasNext={queue.length > 0}
          hasPrev={history.length > 0}
          queue={queue as QueueItem[]}
          onSelectQueueItem={selectQueueItem}
          isQueueItemLiked={(id) => isLiked(id)}
        />
      </View>
    </View>
  );
}

function getDaytimeGreeting() {
  const h = new Date().getHours();
  if (h < 12) return 'morning';
  if (h < 18) return 'afternoon';
  return 'evening';
}

const styles = StyleSheet.create({
  page: {
    flex: 1,
    overflow: 'hidden',
  },
  contentContainer: {
    padding: 24,
    paddingTop: 26,
    paddingBottom: 320, // leave room for floating player expanded and backdrop
    gap: 18,
  },
  bgGlowWrap: {
    position: 'absolute',
    left: -140,
    right: -140,
    top: -160,
    height: 520,
  },
  bgGlow: {
    flex: 1,
    borderRadius: 340,
    transform: [{ rotateZ: '-12deg' }],
  },
  heroCard: {
    borderRadius: 18,
    paddingVertical: 18,
    paddingHorizontal: 16,
    marginBottom: 8,
    borderWidth: 1,
    gap: 6,
    overflow: 'hidden',
  },
  heroCardLight: {
    backgroundColor: 'rgba(255,255,255,0.86)',
    borderColor: 'rgba(0,0,0,0.06)',
    shadowColor: '#000',
    shadowOpacity: 0.07,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 10 },
  },
  heroCardDark: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderColor: 'rgba(255,255,255,0.12)',
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: 16 },
  },
  heroGlow: {
    ...StyleSheet.absoluteFillObject,
  },
  greetingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  greetingText: {
    fontWeight: '800',
    letterSpacing: 0.2,
  },
  greetingActions: {
    flexDirection: 'row',
    gap: 14,
    alignItems: 'center',
  },
  loginPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  loginPillPrimary: {
    backgroundColor: '#1DB954',
  },
  loginPillGhost: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
  },
  loginPillText: {
    color: '#111',
    fontWeight: '700',
  },
  loginPillTextGhost: {
    color: '#fff',
  },
  heroSub: {
    opacity: 0.7,
  },
  helperText: {
    opacity: 0.7,
    marginBottom: 8,
  },
  searchInput: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 10,
    fontSize: 15,
    fontWeight: '600',
    backgroundColor: 'transparent',
    borderWidth: 0,
  },
  searchGradient: {
    borderRadius: 18,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  searchGradientFocused: {
    borderColor: 'rgba(29,185,84,0.55)',
  },
  searchInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: 18,
  },
  searchInnerFocused: {
    shadowColor: '#1DB954',
    shadowOpacity: 0.18,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
  },
  seedsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 10,
  },
  seedChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 20,
    backgroundColor: '#1DB954',
    gap: 6,
  },
  seedText: {
    color: '#fff',
    maxWidth: 140,
  },
  seedRemove: {
    color: '#fff',
    fontWeight: '700',
  },
  actionButton: {
    backgroundColor: '#1DB954',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 8,
  },
  actionButtonDisabled: {
    opacity: 0.5,
  },
  actionButtonText: {
    color: '#fff',
    fontWeight: '700',
  },
  resultRow: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 14,
    padding: 10,
    gap: 10,
    borderWidth: 1,
  },
  resultRowLight: {
    backgroundColor: '#fff',
    borderColor: 'rgba(0,0,0,0.05)',
  },
  resultRowDark: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderColor: 'rgba(255,255,255,0.1)',
  },
  coverThumb: {
    width: 56,
    height: 56,
    borderRadius: 10,
    backgroundColor: '#ccc',
  },
  resultMeta: {
    flex: 1,
    gap: 2,
  },
  resultTitle: {
    fontWeight: '700',
  },
  resultArtist: {
    opacity: 0.8,
  },
  resultAlbum: {
    opacity: 0.6,
    fontSize: 12,
  },
  resultActions: {
    gap: 6,
  },
  searchResultCard: {
    borderRadius: 18,
    padding: 12,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  searchResultCardLight: {
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderColor: 'rgba(0,0,0,0.06)',
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 10 },
  },
  searchResultCardDark: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderColor: 'rgba(255,255,255,0.10)',
  },
  searchResultLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
    paddingRight: 10,
  },
  searchCoverWrap: {
    width: 54,
    height: 54,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: 'rgba(0,0,0,0.06)',
  },
  searchCover: {
    width: '100%',
    height: '100%',
  },
  searchResultMeta: {
    flex: 1,
    gap: 2,
  },
  searchResultActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  iconButtonRound: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconButtonGhost: {
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.14)',
    backgroundColor: 'rgba(255,255,255,0.10)',
  },
  iconButtonRoundPrimary: {
    backgroundColor: '#1DB954',
  },
  iconButtonRoundActive: {
    backgroundColor: 'rgba(29,185,84,0.25)',
    borderWidth: 1,
    borderColor: 'rgba(29,185,84,0.55)',
  },
  iconButtonRoundLikeActive: {
    backgroundColor: '#e11d48',
  },
  iconButtonDisabled: {
    opacity: 0.5,
  },
  likeButton: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.12)',
  },
  likeButtonActive: {
    backgroundColor: '#1DB954',
  },
  bucketHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  carouselContent: {
    gap: 12,
    paddingHorizontal: 14,
  },
  cardItem: {
    borderRadius: 14,
    padding: 10,
    width: 152,
  },
  bucketLabel: {
    fontWeight: '800',
    fontSize: 16,
  },
  bucketSeed: {
    opacity: 0.7,
    fontSize: 12,
  },
  smallButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  smallButtonPrimary: {
    backgroundColor: '#1DB954',
  },
  smallButtonDisabled: {
    backgroundColor: 'rgba(0,0,0,0.12)',
  },
  smallButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  smallButtonGhost: {
    borderWidth: 1,
    borderColor: '#1DB954',
  },
  smallButtonGhostText: {
    color: '#1DB954',
  },
  errorText: {
    color: '#ff6b6b',
    marginTop: 4,
  },
  bucketCover: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 14,
    marginBottom: 6,
  },
  cardImageWrap: {
    position: 'relative',
  },
  likeFloating: {
    position: 'absolute',
    right: 6,
    top: 6,
    padding: 6,
    borderRadius: 999,
  },
  mightLikeHeader: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 10,
  },
  mightLikeTitle: {
    fontWeight: '900',
    fontSize: 16,
    letterSpacing: 0.2,
  },
  mightLikeSub: {
    opacity: 0.7,
    fontSize: 12,
  },
  queueIndex: {
    width: 26,
    alignItems: 'center',
  },
  queueIndexText: {
    opacity: 0.6,
  },
  floatingPlayer: {
    position: 'absolute',
    left: 16,
    right: 16,
    alignItems: 'stretch',
    zIndex: 30,
    elevation: 0,
    shadowColor: '#000',
    shadowOpacity: 0,
    shadowRadius: 0,
    shadowOffset: { width: 0, height: 0 },
    paddingTop: 6,
  },
  playerBackdrop: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.06)',
  },
});
