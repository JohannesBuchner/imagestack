import numpy
import h5py
import matplotlib.pyplot as plt

nx = 301
ny = 201

ni = 20
nj = 20
nk = 20

ni = 12
nj = 12
nk = 12

x = numpy.linspace(-100, 100, nx)
y = numpy.linspace(-100, 100, ny)
X, Y = numpy.meshgrid(x, y)

def genfunc(i, j, k):
	sigmax = 30
	sigmay = 10 + i*30./ni
	slopex  = (0.2 + j*3./nj)/2
	slopey  = (0.2 + k*3./nk)/2
	Z = numpy.exp(-(((X/sigmax)**2)**slopex + ((Y/sigmay)**2)**slopey))
	return Z

def genall():
	for i in range(ni):
		for j in range(nj):
			for k in range(nk):
				yield (i, j, k), genfunc(i,j,k)

def pca(M):
	mean = M.mean(axis=0)
	Moffset = M - mean.reshape((1,-1))
	U, s, Vt = numpy.linalg.svd(Moffset, full_matrices=False)
	V = Vt.T
	print 'variance explained:', s**2/len(M)
	return U, s, V, mean

def pca_predict(U, s, V, mean):
	S = numpy.diag(s)
	return numpy.dot(U, numpy.dot(S, V.T)) + mean.reshape((1,-1))

def pca_get_vectors(s, V, mean):
	#U = numpy.eye(len(s))
	#return pca_predict(U, s, V, mean)
	Sroot = numpy.diag(s**0.5)
	return numpy.dot(Sroot, V.T)

def pca_cut(U, s, V, mean, ncomponents=20):
	return U[:, :ncomponents], s[:ncomponents], V[:,:ncomponents], mean

def pca_check(M, U, s, V, mean):
	# if we use only the first 20 PCs the reconstruction is less accurate
	Mhat2 = pca_predict(U, s, V, mean)
	print "Using %d PCs, MSE = %.6G"  % (len(s), numpy.mean((M - Mhat2)**2))
	return M - Mhat2

if __name__ == '__main__':
	import sys
	FFMPEG_BIN = 'ffmpeg'
	from subprocess import Popen, PIPE
	# store as hdf5
	if sys.argv[1] == 'hdf5':
		with h5py.File('test.hdf5', 'w') as f:
			d = f.create_dataset("images", (ni,nj,nk,nx,ny))
			shape = (1,1,1,nx,ny)
			for params, Z in genall():
				d[params] = Z.reshape(shape)
	
	elif sys.argv[1] == 'hdf5gz':
		with h5py.File('testgz.hdf5', 'w') as f:
			d = f.create_dataset("images", (ni,nj,nk,nx,ny), compression="gzip", compression_opts=9)
			shape = (1,1,1,nx,ny)
			for params, Z in genall():
				d[params] = Z.reshape(shape)
	
	elif sys.argv[1] == 'hdf5gzshuffle':
		with h5py.File('testgzshuffle.hdf5', 'w') as f:
			d = f.create_dataset("images", (ni,nj,nk,nx,ny), compression="gzip", compression_opts=9, shuffle=True)
			shape = (1,1,1,nx,ny)
			for params, Z in genall():
				d[params] = Z.reshape(shape)
	
	elif sys.argv[1] == 'hdf5diff':
		f = h5py.File('testgzshuffle.hdf5', 'r')
		data = f['images'].value.reshape((ni*nj*nk,nx*ny))
		mean = data.mean(axis=0)
		lo = data.min(axis=0)
		hi = data.max(axis=0)
		hidiff = hi.reshape((1,-1)) - data
		lodiff = data - lo.reshape((1,-1))
		from_high = hidiff < lodiff
		diff = numpy.where(from_high, hidiff, lodiff)
		# make levels
		#levels = (255 * ((data - lo.reshape((1,-1))) / (hi - lo).reshape((1,-1)))).astype(uint8)
		# predict level from neighbor pixel
		
		print 'writing...'
		with h5py.File('testdiff.hdf5', 'w') as f:
			d = f.create_dataset("mean", data=mean, compression="gzip", compression_opts=9, shuffle=True)
			d = f.create_dataset("lo", data=lo, compression="gzip", compression_opts=9, shuffle=True)
			d = f.create_dataset("hi", data=hi, compression="gzip", compression_opts=9, shuffle=True)
			d = f.create_dataset("from_high", data=from_high, compression="gzip", compression_opts=9, shuffle=True)
			d = f.create_dataset("diff", data=diff, compression="gzip", compression_opts=9, shuffle=True)
	
	elif sys.argv[1] == 'diffcheck':
		f = h5py.File('testdiff.hdf5', 'r')
		mean = f['mean']
		lo = f['lo']
		hi = f['hi']
		from_high = f['from_high']
		diff = f['diff']
		print 'verifying...'
		for i, (params, Z) in enumerate(genall()):
			value = numpy.where(from_high[i], hi - diff[i,:], lo + diff[i,:]).reshape(Z.shape)
			value = lo + diff[i,:]
			value = value.reshape(Z.shape)
			err = value - Z
			mask = numpy.allclose(value, Z)
			if not mask.all():
				plt.figure("original")
				plt.imshow(Z, cmap='gray', vmin=0, vmax=1)
				plt.figure("recovered")
				plt.imshow(value, cmap='gray', vmin=0, vmax=1)
				plt.figure("diff")
				plt.imshow(Z - value, cmap='gray')
				plt.show()
			assert mask.all(), (mask.sum(), err.sum(), err.size, err.min(), err.max())
		print 'all ok'

	elif sys.argv[1] == 'group':
		f = h5py.File('testgzshuffle.hdf5', 'r')
		data = f['images'].value.reshape((ni*nj*nk,nx*ny))
		mean = data.mean(axis=0)
		resid = data - mean.reshape((1,-1))
		import imagehash
		from PIL import Image
		piles = {}
		print 'hashing ...'
		for i, (params, Z) in enumerate(genall()):
			img = Image.frombytes('L', (nx, ny), (Z*255).astype(numpy.uint8))
			h = str(imagehash.average_hash(img))
			#print i, h
			piles[h] = piles.get(h, []) + [i]
		for pile, items in piles.iteritems():
			print '  %6d %s' % (len(items), pile)
		# for each pile, do pca
		piles = piles.values()
		idx = numpy.zeros(len(data), dtype=numpy.uint8)
		for i, items in enumerate(piles):
			idx[items] = i
		
		print 'writing...'
		with h5py.File('testgroup.hdf5', 'w') as f:
			d = f.create_dataset("index", data=idx, compression="gzip", compression_opts=9, shuffle=True)
			for i, items in enumerate(piles):
				pilemean = data[items,:].mean(axis=0)
				diff = data[items,:] - pilemean.reshape((1,-1))
				f.create_dataset("mean/%d"  % i, data=pilemean, compression="gzip", compression_opts=9, shuffle=True)
				f.create_dataset("pile/%d"  % i, data=diff, compression="gzip", compression_opts=9, shuffle=True)
		
	elif sys.argv[1] == 'img':
		for params, Z in genall():
			plt.imshow(Z)
			plt.title(str(params))
			plt.show()
			plt.clf()
	
	elif sys.argv[1] == 'video':
		# https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
		#ffmpeg -i input -c:v libx264 -crf 0 -preset ultrafast -c:a libmp3lame -b:a 320k output.mp4
		command = [ FFMPEG_BIN,
			'-y', # (optional) overwrite output file if it exists
			'-f', 'rawvideo',
			'-vcodec','rawvideo',
			'-s', '%dx%d' % (nx,ny), # size of one frame
			'-pix_fmt', 'gray16le',
			'-r', '1', # frames per second
			'-an', # Tells FFMPEG not to expect any audio
			'-i', '-', # The imput comes from a pipe
		]
		if sys.argv[2] == 'mp4':
			command += ['-vcodec', 'mpeg4', 'test1.mp4']
		elif sys.argv[2] == 'x264-fast':
			command += ['-c:v', 'libx264', '-crf', '0', 
				'-preset', 'ultrafast', 'test2.mp4']
		elif sys.argv[2] == 'x264-slow':
			command += ['-c:v', 'libx264', '-crf', '0', 
				'-preset', 'veryslow', 'test3.mp4']
		elif sys.argv[2] == 'x264-interm':
			command += ['-c:v', 'libx264', '-crf', '18', 
				'-preset', 'veryfast', 'test4.mp4']
		else:
			raise ValueError('which encoding?')
		
		print command
		pipe = Popen(command, stdin=PIPE)
		for params, Z in genall():
			Zint = (Z * (2**16-1)).astype(numpy.uint16)
			pipe.stdin.write( Zint.flatten().tostring() )
		pipe.stdin.close()
	
	elif sys.argv[1] == 'checkvideo':
		command = [FFMPEG_BIN,
			#'-ss', '00:59;59',
			'-i', sys.argv[2],
			#'-ss', '1',
			'-f', 'image2pipe',
			'-pix_fmt', 'gray16le',
			'-vcodec','rawvideo', '-']
		pipe = Popen(command, stdout=PIPE, bufsize=10**8)
		
		for params, Z in genall():
			Zint = (Z * (2**16-1)).astype(numpy.uint16)
			raw_image = pipe.stdout.read(nx*ny*2)
			image =  numpy.fromstring(raw_image, numpy.uint16)
			image = image.reshape((ny,nx))
			plt.figure("from video")
			plt.imshow(image, vmin=0, vmax=2**16, cmap='gray')
			plt.figure("input")
			plt.imshow(Zint, vmin=0, vmax=2**16, cmap='gray')
			plt.figure("diff")
			diff = Zint - image
			plt.imshow(diff, vmin=0, vmax=2**16, cmap='gray')
			plt.show()
			print diff.shape, Zint.shape, image.shape
			assert (diff==0).all(), ((diff==0).sum(), diff.sum(), diff.size, diff.min(), diff.max())
		print 'all ok'

	elif sys.argv[1] == 'groupvideo':
		import imagehash
		from PIL import Image
		piles = {}
		print 'hashing ...'
		for i, (params, Z) in enumerate(genall()):
			img = Image.frombytes('L', (nx, ny), (Z*255).astype(numpy.uint8))
			hp = str(imagehash.phash(img))
			ha = str(imagehash.average_hash(img))
			hd = str(imagehash.dhash(img))
			print i, hp, ha, hd
			key = (hp, ha, hd)
			piles[key] = piles.get(key, []) + [i]
		
		# for each pile
		keys = sorted(piles.keys())
		# which index each frame came from
		idx = []
		for key in keys:
			items = piles[key]
			print '  %s -- %d items' % (key, len(items))
			for i in items:
				idx.append(i)
		idx = numpy.array(idx)
		# frame number for each index
		loc = numpy.zeros_like(idx)
		for j, i in enumerate(idx):
			loc[i] = j
		# https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
		#ffmpeg -i input -c:v libx264 -crf 0 -preset ultrafast -c:a libmp3lame -b:a 320k output.mp4
		command = [ FFMPEG_BIN,
			'-y', # (optional) overwrite output file if it exists
			'-f', 'rawvideo',
			'-vcodec','rawvideo',
			'-s', '%dx%d' % (nx,ny), # size of one frame
			'-pix_fmt', 'gray16le',
			'-r', '1', # frames per second
			'-an', # Tells FFMPEG not to expect any audio
			'-i', '-', # The imput comes from a pipe
		]
		if sys.argv[2] == 'mp4':
			command += ['-vcodec', 'mpeg4', 'test1group.mp4']
		elif sys.argv[2] == 'x264-fast':
			command += ['-c:v', 'libx264', '-crf', '0', 
				'-preset', 'ultrafast', 'test2group.mp4']
		elif sys.argv[2] == 'x264-slow':
			command += ['-c:v', 'libx264', '-crf', '0', 
				'-preset', 'veryslow', 'test3group.mp4']
		elif sys.argv[2] == 'x264-interm':
			command += ['-c:v', 'libx264', '-crf', '18', 
				'-preset', 'veryfast', 'test4group.mp4']
		else:
			raise ValueError('which encoding?')
		
		numpy.savetxt('groupvideo.txt', loc, fmt='%d')
		print command
		pipe = Popen(command, stdin=PIPE)

		print 'writing...'
		f = h5py.File('testgzshuffle.hdf5', 'r')
		data = f['images'].value.reshape((ni*nj*nk,nx*ny))
		data_resorted = data[idx,:]
		for Zflat in data_resorted:
			Z = Zflat.reshape((nx,ny))
			Zint = (Z * (2**16-1)).astype(numpy.uint16)
			pipe.stdin.write( Zint.flatten().tostring() )
		pipe.stdin.close()

	elif sys.argv[1] == 'pca':
		f = h5py.File('testgzshuffle.hdf5', 'r')
		data = f['images'].value.reshape((ni*nj*nk, nx*ny))
		import sklearn.decomposition
		pca = sklearn.decomposition.PCA(20)
		print('training PCA', data.shape)
		Y = pca.fit_transform(data)
		print(pca.explained_variance_ratio_)
		print(pca.get_params())
		with h5py.File('testpca.hdf5', 'w') as f:
			f.create_dataset("mean", data=pca.mean_, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("components", data=pca.components_, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("values", data=Y, compression="gzip", compression_opts=9, shuffle=True)

	elif sys.argv[1] == 'pcaself':
		f = h5py.File('testgzshuffle.hdf5', 'r')
		data = f['images'].value.reshape((ni*nj*nk, nx*ny))
		import sklearn.decomposition
		print('training PCA', data.shape)
		U, s, V, mean = pca(data)
		pca_check(data, U, s, V, mean)
		with h5py.File('testpcaself.hdf5', 'w') as f:
			f.create_dataset("mean", data=mean, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("components", data=V, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("values", data=s, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("U", data=U, compression="gzip", compression_opts=9, shuffle=True)

		U, s, V, mean = pca_cut(U, s, V, mean, 20)
		pca_check(data, U, s, V, mean)
		with h5py.File('testpcaselfcut.hdf5', 'w') as f:
			f.create_dataset("mean", data=mean, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("components", data=V, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("values", data=s, compression="gzip", compression_opts=9, shuffle=True)
			f.create_dataset("U", data=U, compression="gzip", compression_opts=9, shuffle=True)
	
	elif sys.argv[1] == 'checkpca':
		f = h5py.File(sys.argv[2], 'r')
		mean = f['mean'].value
		V = f['components'].value
		s = f['values'].value
		print s
		U = f['U'].value
		ncomponents = 40
		U, s, V, mean = pca_cut(U, s, V, mean, ncomponents)
		print 'V:', V.shape
		print 'S:', s.shape
		print 'U:', U.shape
		print 'mean:', mean.shape
		
		print 'plotting PCs ...'
		for i, row in enumerate(pca_get_vectors(s, V, mean)):
			if i > 20: break
			Z = row.reshape((ny, nx))
			plt.figure()
			plt.imshow(Z)
			plt.savefig('pca%d.png' % i)
			plt.close()
			
		
		print 'verifying ...'
		f = h5py.File('testgzshuffle.hdf5', 'r')
		data = f['images'].value.reshape((ni*nj*nk, nx*ny))
		predict = pca_predict(U, s, V, mean)
		err = pca_check(data, U, s, V, mean)
		mask = numpy.allclose(err, 0)
		print (mask.sum(), err.sum(), err.size, err.min(), err.max())
		assert mask.all(), (mask.sum(), err.sum(), err.size, err.min(), err.max())
		print 'all ok'
	
	else:
		raise ValueError('unknown command: "%s"' % sys.argv[1])
		
		


