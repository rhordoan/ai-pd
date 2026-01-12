import React, { useEffect, useMemo, useState } from 'react';
import { FlatList, Image, Pressable, StyleSheet, TouchableOpacity, useWindowDimensions, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { ThemedText } from '@/components/themed-text';
import { useColorScheme } from '@/hooks/use-color-scheme';

export type QueueItem = {
  song_id: string;
  title: string;
  artist: string;
  cover_url?: string | null;
};

export type TrackInfo = {
  title: string;
  artist?: string;
  uri?: string;
  songId?: string;
  coverUrl?: string;
};

const SAMPLE_TRACK: TrackInfo = {
  title: 'Sample Mix',
  artist: 'Demo Artist',
  coverUrl: 'https://via.placeholder.com/200x200.png?text=Cover',
};

type Props = {
  track?: TrackInfo | null;
  isLiked?: boolean;
  onToggleLike?: (songId?: string) => void;
  onNext?: () => void;
  onPrev?: () => void;
  hasNext?: boolean;
  hasPrev?: boolean;
  queue?: QueueItem[];
  onSelectQueueItem?: (songId: string) => void;
  isQueueItemLiked?: (songId: string) => boolean;
};

export function MiniPlayer({
  track,
  isLiked = false,
  onToggleLike,
  onNext,
  onPrev,
  hasNext = false,
  hasPrev = false,
  queue = [],
  onSelectQueueItem,
  isQueueItemLiked,
}: Props) {
  const isDark = (useColorScheme() ?? 'light') === 'dark';
  const { height: windowHeight } = useWindowDimensions();
  const [expanded, setExpanded] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const SIM_DURATION = 90_000; // 90s simulated duration
  const activeTrack = track ?? SAMPLE_TRACK;

  // Reset position when track changes
  useEffect(() => {
    setPosition(0);
    // Autoplay when a real track is selected
    setPlaying(!!track);
  }, [track, activeTrack.title, activeTrack.artist]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      setPosition((prev) => {
        const next = prev + 1000;
        if (next >= SIM_DURATION) {
          setPlaying(false);
          return SIM_DURATION;
        }
        return next;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [playing]);

  const duration = SIM_DURATION;
  const progress = duration > 0 ? position / duration : 0;

  const formatted = useMemo(() => {
    const fmt = (ms: number) => {
      const totalSec = Math.floor(ms / 1000);
      const m = Math.floor(totalSec / 60)
        .toString()
        .padStart(1, '0');
      const s = (totalSec % 60).toString().padStart(2, '0');
      return `${m}:${s}`;
    };
    return {
      pos: fmt(position),
      dur: fmt(duration),
    };
  }, [position, duration]);

  const seekBy = (deltaMs: number) => {
    const next = Math.max(0, Math.min(duration, position + deltaMs));
    setPosition(next);
  };

  const toggleExpanded = () => setExpanded((e) => !e);
  const togglePlay = () => setPlaying((p) => !p);

  return (
    <View
      style={[
        styles.container,
        isDark ? styles.containerDark : styles.containerLight,
        expanded && styles.containerExpanded,
        expanded && { maxHeight: Math.min(windowHeight * 0.78, 720) },
      ]}
    >
      {!expanded && (
        <Pressable style={styles.metaRow} accessibilityRole="button" onPress={toggleExpanded}>
          <View style={styles.coverThumbWrap}>
            <Image
              source={{ uri: activeTrack.coverUrl ?? SAMPLE_TRACK.coverUrl! }}
              style={styles.coverThumb}
              resizeMode="cover"
            />
          </View>
          <View style={styles.metaText}>
            <ThemedText numberOfLines={1} style={styles.title}>{activeTrack.title}</ThemedText>
            <ThemedText numberOfLines={1} style={styles.artist}>{activeTrack.artist ?? 'Unknown artist'}</ThemedText>
          </View>

          {onToggleLike ? (
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={isLiked ? 'Unlike' : 'Like'}
              onPress={(e) => {
                // prevent toggling expanded when tapping the heart
                // @ts-expect-error stopPropagation exists in RN events
                e?.stopPropagation?.();
                onToggleLike(activeTrack.songId);
              }}
              style={[styles.likeButton, isLiked && styles.likeButtonActive]}
            >
              <Ionicons name={isLiked ? 'heart' : 'heart-outline'} size={16} color={isLiked ? '#fff' : '#111'} />
            </Pressable>
          ) : null}

          <Pressable
            accessibilityRole="button"
            accessibilityLabel={playing ? 'Pause' : 'Play'}
            onPress={(e) => {
              // prevent toggling expanded when tapping play
              // @ts-expect-error stopPropagation exists in RN events
              e?.stopPropagation?.();
              togglePlay();
            }}
            style={styles.playInline}
          >
            <Ionicons name={playing ? 'pause' : 'play'} size={18} color="#111" />
          </Pressable>

          <Ionicons name="chevron-up" size={16} color={isDark ? '#fff' : '#111'} />
        </Pressable>
      )}

      {expanded && (
        <View style={styles.expandedWrap}>
          <View style={styles.expandedHeader}>
            <View style={styles.grabber} />
            <TouchableOpacity
              accessibilityRole="button"
              accessibilityLabel="Close player"
              onPress={() => setExpanded(false)}
              style={styles.closeButton}
            >
              <Ionicons name="chevron-down" size={18} color={isDark ? '#fff' : '#111'} />
            </TouchableOpacity>
          </View>

          <View style={styles.heroRow}>
            <View style={styles.coverHeroWrap}>
              <Image
                source={{ uri: activeTrack.coverUrl ?? SAMPLE_TRACK.coverUrl! }}
                style={styles.coverHero}
                resizeMode="cover"
              />
            </View>
          </View>

          <View style={styles.expandedMeta}>
            <View style={{ flex: 1 }}>
              <ThemedText numberOfLines={1} style={styles.expandedTitle}>{activeTrack.title}</ThemedText>
              <ThemedText numberOfLines={1} style={styles.expandedArtist}>{activeTrack.artist ?? 'Unknown artist'}</ThemedText>
            </View>
            {onToggleLike ? (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={isLiked ? 'Unlike' : 'Like'}
                onPress={() => onToggleLike(activeTrack.songId)}
                style={[styles.likeButtonBig, isLiked && styles.likeButtonBigActive]}
              >
                <Ionicons name={isLiked ? 'heart' : 'heart-outline'} size={18} color={isLiked ? '#fff' : (isDark ? '#fff' : '#111')} />
              </Pressable>
            ) : null}
          </View>

          <View style={styles.progressWrap}>
            <View style={styles.progressTrack}>
              <View style={[styles.progressBar, { width: `${Math.min(100, progress * 100)}%` }]} />
            </View>
            <View style={styles.timeRow}>
              <ThemedText style={styles.timeText}>{formatted.pos}</ThemedText>
              <ThemedText style={styles.timeText}>{formatted.dur}</ThemedText>
            </View>
          </View>
          <View style={styles.controlsRow}>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Rewind 15 seconds"
              onPress={() => seekBy(-15000)}
              style={styles.iconButton}
              disabled={!duration}
            >
              <Ionicons name="play-back-outline" size={22} color={isDark ? '#fff' : '#111'} />
            </Pressable>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Previous"
              onPress={onPrev}
              style={[styles.iconButton, !hasPrev && styles.iconButtonDisabled]}
              disabled={!hasPrev}
            >
              <Ionicons name="play-skip-back" size={22} color={isDark ? '#fff' : '#111'} />
            </Pressable>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel={playing ? 'Pause' : 'Play'}
              onPress={togglePlay}
              style={styles.playButton}
            >
              <Ionicons name={playing ? 'pause' : 'play'} size={22} color={isDark ? '#111' : '#fff'} />
            </Pressable>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Next"
              onPress={onNext}
              style={[styles.iconButton, !hasNext && styles.iconButtonDisabled]}
              disabled={!hasNext}
            >
              <Ionicons name="play-skip-forward" size={22} color={isDark ? '#fff' : '#111'} />
            </Pressable>

            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Forward 15 seconds"
              onPress={() => seekBy(15000)}
              style={styles.iconButton}
              disabled={!duration}
            >
              <Ionicons name="play-forward-outline" size={22} color={isDark ? '#fff' : '#111'} />
            </Pressable>
          </View>

          <View style={[styles.queuePanel, isDark ? styles.queuePanelDark : styles.queuePanelLight]}>
            <View style={styles.queueHeader}>
              <ThemedText style={styles.queueTitle}>Up next</ThemedText>
              <ThemedText style={styles.queueCount}>{queue.length} songs</ThemedText>
            </View>

            {queue.length === 0 ? (
              <ThemedText style={styles.queueEmpty}>Queue is empty. Play a song to generate recommendations.</ThemedText>
            ) : (
              <FlatList
                data={queue}
                keyExtractor={(it) => it.song_id}
                scrollEnabled
                nestedScrollEnabled
                showsVerticalScrollIndicator={false}
                style={[
                  styles.queueList,
                  { maxHeight: Math.min(windowHeight * 0.30, 280) },
                  ({ scrollbarWidth: 'none' } as any),
                ]}
                contentContainerStyle={{ gap: 10, paddingBottom: 4 }}
                renderItem={({ item, index }) => {
                  const liked = isQueueItemLiked ? isQueueItemLiked(item.song_id) : false;
                  return (
                    <TouchableOpacity
                      activeOpacity={0.9}
                      style={[styles.queueRow, isDark ? styles.queueRowDark : styles.queueRowLight]}
                      onPress={() => onSelectQueueItem?.(item.song_id)}
                    >
                      <ThemedText style={styles.queueIndex}>{index + 1}</ThemedText>
                      <View style={styles.queueCoverWrap}>
                        <Image
                          source={{ uri: item.cover_url ?? SAMPLE_TRACK.coverUrl! }}
                          style={styles.queueCover}
                          resizeMode="cover"
                        />
                      </View>
                      <View style={{ flex: 1 }}>
                        <ThemedText numberOfLines={1} style={styles.queueRowTitle}>{item.title}</ThemedText>
                        <ThemedText numberOfLines={1} style={styles.queueRowArtist}>{item.artist}</ThemedText>
                      </View>
                      {onToggleLike ? (
                        <Pressable
                          accessibilityRole="button"
                          accessibilityLabel={liked ? 'Unlike' : 'Like'}
                          onPress={(e) => {
                            // prevent selecting queue item when tapping heart
                            // @ts-expect-error stopPropagation exists in RN events
                            e?.stopPropagation?.();
                            onToggleLike(item.song_id);
                          }}
                          style={[styles.queueLike, liked && styles.queueLikeActive]}
                        >
                          <Ionicons name={liked ? 'heart' : 'heart-outline'} size={16} color={liked ? '#fff' : (isDark ? '#fff' : '#111')} />
                        </Pressable>
                      ) : null}
                    </TouchableOpacity>
                  );
                }}
              />
            )}
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 10,
    gap: 10,
  },
  containerLight: {
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 8 },
  },
  containerDark: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 12 },
  },
  containerExpanded: {
    paddingTop: 12,
    paddingBottom: 14,
    overflow: 'hidden',
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  coverThumbWrap: {
    width: 44,
    height: 44,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: 'rgba(0,0,0,0.08)',
  },
  coverThumb: {
    width: '100%',
    height: '100%',
  },
  metaText: {
    flex: 1,
  },
  title: {
    fontWeight: '700',
  },
  artist: {
    opacity: 0.7,
    fontSize: 12,
  },
  likeButton: {
    padding: 6,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.12)',
    marginRight: 6,
  },
  likeButtonActive: {
    backgroundColor: '#1DB954',
    borderColor: '#1DB954',
  },
  playInline: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1DB954',
  },
  expandedWrap: {
    gap: 10,
    flex: 1,
    minHeight: 0,
  },
  expandedHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  grabber: {
    height: 4,
    width: 42,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.18)',
    alignSelf: 'center',
    flex: 1,
    maxWidth: 56,
  },
  closeButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  heroRow: {
    alignItems: 'center',
    marginTop: 2,
  },
  coverHeroWrap: {
    width: 210,
    height: 210,
    borderRadius: 18,
    overflow: 'hidden',
    backgroundColor: 'rgba(0,0,0,0.10)',
  },
  coverHero: {
    width: '100%',
    height: '100%',
  },
  expandedMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 10,
    marginBottom: 2,
    gap: 10,
  },
  expandedTitle: {
    fontSize: 16,
    fontWeight: '800',
  },
  expandedArtist: {
    opacity: 0.75,
    fontSize: 12,
  },
  likeButtonBig: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.18)',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  likeButtonBigActive: {
    backgroundColor: '#1DB954',
    borderColor: '#1DB954',
  },
  progressWrap: {
    gap: 8,
  },
  progressTrack: {
    height: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.15)',
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#1DB954',
  },
  timeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  timeText: {
    opacity: 0.7,
    fontSize: 12,
  },
  controlsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  iconButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  iconButtonDisabled: {
    opacity: 0.4,
  },
  playButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1DB954',
  },
  playButtonDisabled: {
    opacity: 0.6,
  },
  queuePanel: {
    marginTop: 12,
    borderRadius: 18,
    padding: 12,
    borderWidth: 1,
    flex: 1,
    minHeight: 0,
    overflow: 'hidden',
  },
  queuePanelLight: {
    backgroundColor: 'rgba(255,255,255,0.85)',
    borderColor: 'rgba(0,0,0,0.06)',
  },
  queuePanelDark: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderColor: 'rgba(255,255,255,0.10)',
  },
  queueHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  queueTitle: {
    fontWeight: '800',
    fontSize: 14,
  },
  queueCount: {
    opacity: 0.7,
    fontSize: 12,
  },
  queueEmpty: {
    opacity: 0.7,
    fontSize: 12,
  },
  queueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderRadius: 14,
    borderWidth: 1,
  },
  queueRowLight: {
    backgroundColor: 'rgba(255,255,255,0.92)',
    borderColor: 'rgba(0,0,0,0.06)',
  },
  queueRowDark: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderColor: 'rgba(255,255,255,0.10)',
  },
  queueIndex: {
    width: 18,
    opacity: 0.6,
    fontSize: 12,
  },
  queueCoverWrap: {
    width: 42,
    height: 42,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: 'rgba(0,0,0,0.08)',
  },
  queueCover: {
    width: '100%',
    height: '100%',
  },
  queueRowTitle: {
    fontWeight: '700',
  },
  queueRowArtist: {
    opacity: 0.7,
    fontSize: 12,
  },
  queueLike: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.18)',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  queueLikeActive: {
    backgroundColor: '#e11d48',
    borderColor: '#e11d48',
  },
  queueList: {
    flexGrow: 0,
  },
});
