import React, { useMemo, useState } from 'react';
import { ActivityIndicator, KeyboardAvoidingView, Platform, Pressable, StyleSheet, TextInput, TouchableOpacity, View } from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import { ThemedText } from '@/components/themed-text';
import { useAuth } from '@/providers/auth';
import { useColorScheme } from '@/hooks/use-color-scheme';

export default function LoginScreen() {
  const isDark = (useColorScheme() ?? 'light') === 'dark';
  const { login, signup } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focused, setFocused] = useState<'username' | 'password' | null>(null);
  const router = useRouter();

  const submit = async () => {
    setError(null);
    if (!username.trim() || !password.trim()) return setError('Enter a username and password.');
    if (username.trim().length < 3) return setError('Username must be at least 3 characters.');
    if (password.length < 4) return setError('Password must be at least 4 characters.');
    setLoading(true);
    try {
      if (mode === 'login') {
        await login(username.trim(), password);
      } else {
        await signup(username.trim(), password);
      }
      router.replace('/(tabs)');
    } catch (err) {
      setError((err as Error).message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const title = mode === 'login' ? 'Welcome back' : 'Create your account';
  const subtitle =
    mode === 'login'
      ? 'Sign in to sync your taste and get fresh recs.'
      : 'Join to save likes, build your vibe, and get better recs.';

  const palette = useMemo(() => {
    const bg: readonly [string, string, string] = isDark
      ? ['#050606', '#0a0f0c', '#050606']
      : ['#060707', '#0a0f0c', '#060707'];
    const glass = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.08)';
    const stroke = isDark ? 'rgba(255,255,255,0.12)' : 'rgba(255,255,255,0.14)';
    const text = '#ffffff';
    const muted = 'rgba(255,255,255,0.70)';
    const muted2 = 'rgba(255,255,255,0.55)';
    return { bg, glass, stroke, text, muted, muted2 };
  }, [isDark]);

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
          <View style={styles.brandRow}>
            <View style={styles.brandDot} />
            <ThemedText style={[styles.brand, { color: palette.text }]}>Spotify Redux</ThemedText>
          </View>

          <ThemedText style={[styles.heroTitle, { color: palette.text }]}>{title}</ThemedText>
          <ThemedText style={[styles.heroSub, { color: palette.muted }]}>{subtitle}</ThemedText>

          <View style={[styles.segmentWrap, { backgroundColor: palette.glass, borderColor: palette.stroke }]}>
            <Pressable
              style={[styles.segment, mode === 'login' && styles.segmentActive]}
              onPress={() => setMode('login')}
              accessibilityRole="button"
              accessibilityLabel="Switch to login"
            >
              <ThemedText style={[styles.segmentText, { color: palette.muted }, mode === 'login' && styles.segmentTextActive]}>
                Log in
              </ThemedText>
            </Pressable>
            <Pressable
              style={[styles.segment, mode === 'signup' && styles.segmentActive]}
              onPress={() => setMode('signup')}
              accessibilityRole="button"
              accessibilityLabel="Switch to signup"
            >
              <ThemedText style={[styles.segmentText, { color: palette.muted }, mode === 'signup' && styles.segmentTextActive]}>
                Sign up
              </ThemedText>
            </Pressable>
          </View>

          <View style={[styles.card, { backgroundColor: palette.glass, borderColor: palette.stroke }]}>
            <View style={[styles.field, focused === 'username' && styles.fieldFocused, { borderColor: palette.stroke }]}>
              <Ionicons name="person-outline" size={18} color={palette.muted2} />
              <TextInput
                placeholder="Username"
                placeholderTextColor={palette.muted2}
                autoCapitalize="none"
                autoCorrect={false}
                style={[styles.input, { color: palette.text }, ({ outlineStyle: 'none' } as any)]}
                value={username}
                onChangeText={setUsername}
                onFocus={() => setFocused('username')}
                onBlur={() => setFocused(null)}
                returnKeyType="next"
              />
            </View>

            <View style={[styles.field, focused === 'password' && styles.fieldFocused, { borderColor: palette.stroke }]}>
              <Ionicons name="lock-closed-outline" size={18} color={palette.muted2} />
              <TextInput
                placeholder="Password"
                placeholderTextColor={palette.muted2}
                secureTextEntry={!showPassword}
                style={[styles.input, { color: palette.text }, ({ outlineStyle: 'none' } as any)]}
                value={password}
                onChangeText={setPassword}
                onFocus={() => setFocused('password')}
                onBlur={() => setFocused(null)}
                returnKeyType="done"
                onSubmitEditing={submit}
              />
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={showPassword ? 'Hide password' : 'Show password'}
                onPress={() => setShowPassword((s) => !s)}
                style={styles.eyeButton}
                hitSlop={10}
              >
                <Ionicons name={showPassword ? 'eye-off-outline' : 'eye-outline'} size={18} color={palette.muted2} />
              </Pressable>
            </View>

            {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}

            <Pressable
              accessibilityRole="button"
              accessibilityLabel={mode === 'login' ? 'Log in' : 'Create account'}
              disabled={loading}
              onPress={submit}
              style={({ pressed }) => [styles.ctaWrap, (pressed && !loading) && styles.ctaPressed, loading && styles.ctaDisabled]}
            >
              <LinearGradient
                colors={['#1DB954', '#18a34a']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.cta}
              >
                {loading ? (
                  <ActivityIndicator color="#041007" />
                ) : (
                  <View style={styles.ctaInner}>
                    <ThemedText style={styles.ctaText}>{mode === 'login' ? 'Log in' : 'Create account'}</ThemedText>
                    <Ionicons name="arrow-forward" size={18} color="#041007" />
                  </View>
                )}
              </LinearGradient>
            </Pressable>

            <View style={styles.actionsRow}>
              <TouchableOpacity accessibilityRole="button" onPress={() => router.back()} style={styles.ghostBtn}>
                <ThemedText style={[styles.ghostText, { color: palette.muted }]}>Continue without account</ThemedText>
              </TouchableOpacity>
              <TouchableOpacity
                accessibilityRole="button"
                onPress={() => setMode((m) => (m === 'login' ? 'signup' : 'login'))}
                style={styles.linkBtn}
              >
                <ThemedText style={styles.linkText}>
                  {mode === 'login' ? "Need an account? Sign up" : 'Have an account? Log in'}
                </ThemedText>
              </TouchableOpacity>
            </View>
          </View>
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
    justifyContent: 'center',
  },
  container: {
    width: '100%',
    maxWidth: 520,
    alignSelf: 'center',
    paddingHorizontal: 22,
    paddingVertical: 28,
    gap: 12,
  },
  bgGlowWrap: {
    position: 'absolute',
    left: -120,
    right: -120,
    top: -140,
    height: 520,
  },
  bgGlow: {
    flex: 1,
    borderRadius: 340,
    transform: [{ rotateZ: '-12deg' }],
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  brandDot: {
    width: 10,
    height: 10,
    borderRadius: 999,
    backgroundColor: '#1DB954',
  },
  brand: {
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 0.5,
    opacity: 0.9,
  },
  heroTitle: {
    fontSize: 32,
    fontWeight: '900',
    letterSpacing: -0.4,
  },
  heroSub: {
    fontSize: 14,
    lineHeight: 20,
  },
  segmentWrap: {
    flexDirection: 'row',
    borderRadius: 999,
    borderWidth: 1,
    padding: 4,
    marginTop: 10,
  },
  segment: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
  },
  segmentActive: {
    backgroundColor: 'rgba(29,185,84,0.22)',
  },
  segmentText: {
    fontWeight: '800',
    fontSize: 13,
  },
  segmentTextActive: {
    color: '#fff',
  },
  card: {
    borderRadius: 20,
    padding: 16,
    gap: 12,
    borderWidth: 1,
    marginTop: 10,
  },
  field: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    borderRadius: 14,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 12,
    backgroundColor: 'rgba(0,0,0,0.20)',
  },
  fieldFocused: {
    borderColor: 'rgba(29,185,84,0.65)',
    backgroundColor: 'rgba(29,185,84,0.06)',
  },
  input: {
    flex: 1,
    fontSize: 15,
    paddingVertical: 0,
  },
  eyeButton: {
    paddingLeft: 6,
    paddingVertical: 2,
  },
  errorText: {
    color: '#fb7185',
    fontWeight: '600',
    fontSize: 13,
    marginTop: 2,
  },
  ctaWrap: {
    borderRadius: 16,
    overflow: 'hidden',
    marginTop: 6,
    shadowColor: '#000',
    shadowOpacity: 0.22,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 12 },
  },
  ctaPressed: {
    transform: [{ scale: 0.99 }],
  },
  ctaDisabled: {
    opacity: 0.7,
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
    letterSpacing: 0.3,
  },
  actionsRow: {
    marginTop: 8,
    gap: 10,
  },
  ghostBtn: {
    alignItems: 'center',
    paddingVertical: 8,
  },
  ghostText: {
    fontWeight: '700',
    fontSize: 13,
  },
  linkBtn: {
    alignItems: 'center',
    paddingVertical: 6,
  },
  linkText: {
    color: '#1DB954',
    fontWeight: '800',
  },
});
